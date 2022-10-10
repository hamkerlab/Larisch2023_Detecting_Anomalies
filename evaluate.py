import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
from  matplotlib.colors import LogNorm
import numpy as np
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, accuracy_score
from matplotlib.ticker import LogFormatter 


from cct import create_cct_model
from transformers import BertTokenizer
from misc import *
from tqdm import tqdm

import os
import tensorflow as tf
import pandas as pd
import tensorflow_addons as tfa
from custom_loader import deTokenize

def main():

    print('Start evaluation')
    tf.random.set_seed(314)

    checkpoint_path = "./model/cct_model.hdf5"

    # for colors!
    norm = mp.colors.Normalize(vmin=0, vmax=8)
    cmap = mp.cm.get_cmap('viridis')


    ## Hyperparameters and constants
    positional_emb = True
    conv_layers = 2
    projection_dim = 64
    conv_output_channels = [128,projection_dim]

    num_heads = 5
    transformer_units= [projection_dim,projection_dim]    
    transformer_layers = 5

    learning_rate = 0.0001
    weight_decay = 0.00001

    windows_size = 20

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    ids = bert_tokenizer(bert_tokenizer.mask_token)
    ids = ids['input_ids']## should be something like [101,103,102] -> 101: Start token / 102: End token
    mask_id = ids[1]
    print('Mask ID: ',mask_id)
    ##load the preprocessed Dataset
    x_test = np.load('/scratch/laren/deepL/data/BGL_masked_Xtest_conv5_4_uniq.npy')
    labels = np.load('/scratch/laren/deepL/data/BGL_masked_Ytest_conv5_4_uniq.npy')

    #print(bert_tokenizer.decode([101,103,102]))
    #print(bert_tokenizer.decode([-1,0,1]))
        
    n_sampl, n_ws, pad = np.shape(x_test)
    mask_test = np.zeros((n_sampl, n_ws, pad))
    for i in range(n_sampl):

        sample = x_test[i]
        idx_s, idx_t = np.where(sample>0)
        mask_test[i, idx_s, idx_t] = 1

    plt.figure()
    plt.hist(np.sum(mask_test[:,-1],axis=1))
    plt.savefig('hist_mask')


    x_test = np.expand_dims(x_test, -1)
    x_test = x_test.astype('int32')

    mask_test = np.expand_dims(mask_test, -1)
    mask_test = mask_test.astype('int32')

    y_test = np.copy(x_test)
    y_test = y_test[:,-1,:]
    x_test = np.concatenate((x_test,mask_test),-1)
    
    inpt_shape = np.shape(x_test)[1:]

    # load the model
    cct_model = create_cct_model(
                image_size = windows_size, 
                input_shape = inpt_shape, 
                num_classes = bert_tokenizer.vocab_size,
                num_heads = num_heads,
                num_conv_layers = conv_layers,
                num_output_channels = conv_output_channels,  
                projection_dim = projection_dim, 
                transformer_units = transformer_units, 
                transformer_layers = transformer_layers,
                positional_emb = positional_emb)
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    cct_model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.SparseCategoricalCrossentropy()],)

    cct_model.load_weights(checkpoint_path)
    print(cct_model.summary())

    print(np.shape(x_test), np.shape(y_test), np.shape(labels))

    good_true = np.where(labels == 0)[0]
    bad_true = np.where(labels == 1)[0]

    """    
    predicts = cct_model.predict(x_test[bad_true[0:10]])
    print(np.shape(predicts))
    plt.figure()
    plt.hist(np.ndarray.flatten(predicts[0,0]))
    plt.savefig('test_softmax_bad')
 
    """
    print(bad_true[0:10])
    #for i in range(len(bad_true[0:10])):
    #    print(predicts[i])    
    #    print(y_test[bad_true[i]])
    #    print('--------------')


    ## because of memory issues, make predictions in 1000' chunks
    chunk_size = 750
    n_samples = len(x_test)

    n_chunks = n_samples//chunk_size
    samples_left = n_samples - (n_chunks*chunk_size)
    
    ### save the logit scores of the input tokens
    all_predictions = np.zeros((n_samples,pad))
    ### get chunk-wise through the test set
    ### predict for each sample in the chunk the sequence
    for c in tqdm(range(n_chunks+1),ncols=80): # iterate over all *c*hunks #n_chunks+1
        if c < n_chunks:
            act_X = np.copy(x_test[0+(c*chunk_size):chunk_size+(c*chunk_size)])
            chunk_predict = np.zeros((chunk_size, pad))
        else: # the last remainig samples
            act_X = np.copy(x_test[-samples_left:])
            chunk_predict = np.zeros((samples_left, pad))
        act_Y = np.squeeze(act_X[:,-1,:,0])
        predict = cct_model.predict(act_X)
        for s in range(len(act_Y)):# iterate over all *s*amples in the current chunk #
            ## get the actual log line
            act_y = act_Y[s,act_Y[s]>0]
            ## look where is the end token (102)
            end_idx = np.where( act_y == 102)[0][0]
            ## create a list of important indizes 
            list_idx = np.linspace(0, end_idx,end_idx+1,dtype='int16')
            ## get only the important predictions
            act_pred = predict[s,act_Y[s]>0]
            comp = act_pred[list_idx,act_y[list_idx]]
            #comp = [ act_pred[j,act_y[j]] for j in range(len(act_y)) ]
            chunk_predict[s,0:len(act_y)] = comp
        ## save the predicted chunk
        if c < n_chunks:
            all_predictions[0+(c*chunk_size):chunk_size+(c*chunk_size)] = chunk_predict
        else:
            all_predictions[-samples_left:] = chunk_predict


    np.save('x_test_Predictions_uniq.npy',all_predictions)

class MidPointLogNorm(LogNorm):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        LogNorm.__init__(self, vmin=vmin, vmax=vmax, clip=clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        x,y = [np.log(self.vmin), np.log(self.midpoint), np.log(self.vmax)], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(np.log(value), x, y))

def measure():

    x_test = np.load('/scratch/laren/deepL/data/BGL_masked_Xtest_conv5_4_uniq.npy')
    labels = np.load('/scratch/laren/deepL/data/BGL_masked_Ytest_conv5_4_uniq.npy')

    x_logit = np.load('x_test_Predictions_uniq.npy')

    y_test = np.copy(x_test)
    y_test = y_test[:,-1,:]
    y_test = np.squeeze(y_test)
    #y_test = y_test[0:1000]
    #labels = labels[0:1000]

    if not os.path.exists('reports_uniq'):
        os.mkdir('reports_uniq')

    top_g = 5
    thresh_list = np.linspace(1e-3,1e-5,top_g)
    
    print(np.shape(x_logit), np.shape(y_test))
    n_samples, padd = np.shape(x_logit)
    #print(x_logit[0])
    for t in range(5):
        all_predicts = np.copy(y_test)
        ## if the logit score is below the threshold, set it to -1
        idx_th = np.where((x_logit < thresh_list[t]) & (x_logit > 0.0) )
        all_predicts[idx_th] = -1           
        n_samples, n_words = np.shape(y_test)

        for h in range(1,4):
            good_idx = []
            bad_idx = []
            thresh = h
                
            print('Look for anomalies')

            for i in tqdm(range(n_samples),ncols=80):
                ## look only at tokens between the start and endtoken
                end_idx = np.where( y_test[i] == 102)[0][0]
                actual_pred = all_predicts[i]
                actual_pred = actual_pred[1:end_idx]
                idx_m = np.where(actual_pred == -1)[0]
                if len(idx_m) < thresh:
                    good_idx.append(i)
                    #print('good one')
                    #print(y_test[i])
                    #print(all_predicts[i])
                    #print('---------------------')
                else:
                    #print('bad one')
                    #print(i)
                    #print(y_test[i])
                    #print(all_predicts[i])
                    #print('---------------------')
                    bad_idx.append(i)

            good_true = np.where(labels == 0)[0]
            bad_true = np.where(labels == 1)[0]

            print('Good predicted = ', len(good_idx),'  Good true = ', len(good_true))
            print('Bad predicted = ', len(bad_idx),'  Bad true = ', len(bad_true))


            p_lab = np.zeros(n_samples)
            p_lab[bad_idx] = 1
                
            #print(classification_report(labels,p_lab))
            report = pd.DataFrame(classification_report(labels,p_lab, output_dict=True))
            report.to_csv('reports_uniq/class_report_short_prob_t%i_h%i.csv'%(t,h))
    

    ### recreate some of the input sequences 
    
    print(np.shape(x_logit), np.shape(labels))
    print(y_test[0])
    start_t = 101
    end_t = 102
    print(deTokenize(y_test[0], start_t, end_t))
    print(x_logit[0,1:9])

    good_true = np.where(labels == 0)[0]
    np.random.shuffle(good_true)
    bad_true = np.where(labels == 1)[0]
    np.random.shuffle(bad_true)
    n_samples = 5

    good_samples = good_true[0:n_samples]
    
    bad_samples = bad_true[0:n_samples]
   

    cmap = plt.get_cmap('RdYlGn')
    norm = MidPointLogNorm(vmin= thresh_list[0]/5, vmax=1.15, midpoint=thresh_list[0]) # mp.colors.TwoSlopeNorm(thresh_list[0],0.0, 1.15)

    plt.figure(figsize=(6,4))
    for n in range(n_samples):
        act_y = y_test[good_samples[n]]
        act_y = act_y[act_y>0]
        act_string = deTokenize(y_test[good_samples[n]], start_t, end_t)
        comp = x_logit[good_samples[n]]
        comp = comp[y_test[good_samples[n]]>0]
        comp = comp[1:-1] # throw the first and last token away
        y_pos = n_samples-n
        plt.text(0, y_pos,act_string[0], backgroundcolor = cmap(norm(comp[0])),fontsize=15 )
        x_pos = len(act_string[0])
        for s in range(1,len(act_string)):
            plt.text(x_pos, y_pos,act_string[s], backgroundcolor = cmap(norm(comp[s])),fontsize=15 )
            x_pos += len(act_string[s])
    plt.axis('off') 
    plt.ylim(0,n_samples)
    plt.xlim(0,x_pos)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm = norm)
    sm.set_array([])
    formatter = LogFormatter(10, labelOnlyBase=False) 
    #cb = mp.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    plt.colorbar(sm, location='bottom', orientation='horizontal', shrink = 0.75, anchor=(0.5, .9),pad=0.005) 
    plt.savefig('good_strings_uniq',bbox_inches='tight')


    plt.figure(figsize=(6,4))
    for n in range(n_samples):
        act_y = y_test[bad_samples[n]]
        act_y = act_y[act_y>0]
        act_string = deTokenize(y_test[bad_samples[n]], start_t, end_t)
        comp = x_logit[bad_samples[n]]
        comp = comp[y_test[bad_samples[n]]>0]
        comp = comp[1:-1] # throw the first and last token away
        y_pos = n_samples-n-0.5
        plt.text(0,y_pos,act_string[0], backgroundcolor = cmap(norm(comp[0])),fontsize=15 )
        x_pos = len(act_string[0])
        for s in range(1,len(act_string)):
            plt.text(x_pos,y_pos,act_string[s], backgroundcolor = cmap(norm(comp[s])),fontsize=15 )
            x_pos += len(act_string[s])
    plt.axis('off') 
    plt.ylim(0,n_samples)
    plt.xlim(0,x_pos)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm = norm)
    sm.set_array([])
    #cb = mp.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    plt.colorbar(sm, location='bottom', orientation='horizontal', shrink = 0.5, anchor=(0.5, .9),pad=0.005)   
    plt.savefig('bad_strings_uniq',bbox_inches='tight')


if __name__ == "__main__":
    main()
    measure()


