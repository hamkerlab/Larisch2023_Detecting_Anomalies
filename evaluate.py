import matplotlib as mp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from cct import create_cct_model
from transformers import BertTokenizer

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from custom_loader import deTokenize
from delete_redundant import compare
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from tqdm import tqdm

import os

class MidPointLogNorm(LogNorm):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        LogNorm.__init__(self, vmin=vmin, vmax=vmax, clip=clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        x,y = [np.log(self.vmin), np.log(self.midpoint), np.log(self.vmax)], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(np.log(value), x, y))

def get_logits():
        
    tf.random.set_seed(314)

    checkpoint_path = "./model/cct_model.hdf5"

    ## Hyperparameters and constants for the CCT
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

    ## load tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    ids = bert_tokenizer(bert_tokenizer.mask_token)
    ids = ids['input_ids']## should be something like [101,103,102] -> 101: Start token / 102: End token
    mask_id = ids[1]
    print('Mask ID: ',mask_id)

    ##load the preprocessed Dataset
    #### check if it exists. If not, create it
    if os.path.isfile('./data/BGL_masked_Xtest_uniq.npy'):
        print('Test dataset already prepared')
        x_test = np.load('./data/BGL_masked_Xtest_uniq.npy')
        labels = np.load('./data/BGL_masked_Ytest_uniq.npy')
    else:
        compare() # remove samples from the test dataset, which are in the train dataset
        x_test = np.load('./data/BGL_masked_Xtest_uniq.npy')
        labels = np.load('./data/BGL_masked_Ytest_uniq.npy')

    ### create a mask to mask unimportant parts in the input
    n_sampl, n_ws, pad = np.shape(x_test)
    mask_test = np.zeros((n_sampl, n_ws, pad))
    for i in range(n_sampl):
        sample = x_test[i]
        idx_s, idx_t = np.where(sample>0)
        mask_test[i, idx_s, idx_t] = 1

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

    # print the summary of the loaded CCT model
    #print(cct_model.summary())


    #####
    # Take the test set and process it chunk wise with the CCT
    # Save the predicted logit scores for each samples in the test set
    ####
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


    np.save('./work/x_test_Predictions_uniq.npy',all_predictions)


def createValidationTest():

    ### check if logit scores of test set already exists
    if not os.path.isfile('./work/x_test_Predictions_uniq.npy'):
        ## if not, get the logit scores
        get_logits()

    ## load test data and corresponding logit scores 
    x_test = np.load('./data/BGL_masked_Xtest_uniq.npy')
    labels = np.load('./data/BGL_masked_Ytest_uniq.npy')
    x_logit = np.load('./work/x_test_Predictions_uniq.npy')

    ## split with 20% val and 80% test

    n_samples = len(x_test)
    val_samp = int(n_samples*0.2)
    test_samp = int(n_samples*0.8)

    ## validation set
    val_X = x_test[0:val_samp]
    val_log = x_logit[0:val_samp]
    val_Y = labels[0:val_samp]
        
    ## test set
    test_X = x_test[val_samp:]
    test_log = x_logit[val_samp:]
    test_Y = labels[val_samp:]

    ### check for identical samples between validation and test set 
    ### and sort identical pairs out of the testset
    identical_sum = np.sum(( val_X[0] == val_X[0])*1)
    for i in tqdm(range(len(val_X)), ncols=80):
        compare = ( val_X[i] == test_X)*1
        compare = np.asarray(compare)
        sum_identicals = np.sum(np.sum(compare, axis=1),axis=1)
        idx_identicals = np.where(sum_identicals == identical_sum )[0]
        idx_notIdentical = np.where(sum_identicals != identical_sum )[0]
        if len(idx_identicals) > 0:
            new_testX    = np.copy(test_X[idx_notIdentical])
            new_test_log = np.copy(test_log[idx_notIdentical])
            new_test_Y   = np.copy(test_Y[idx_notIdentical])
        else:
            new_testX = np.copy(test_X) # nothing to do
            new_test_log = np.copy(test_log)
            new_test_Y = np.copy(test_Y)
        ## set the new values
        test_X   = np.asarray(new_testX)
        test_log = np.asarray(new_test_log)
        test_Y   = np.asarray(new_test_Y)


    ### save the final validation and test set
    np.save('./work/val_X',val_X)
    np.save('./work/val_log',val_log)
    np.save('./work/val_Y',val_Y)
    np.save('./work/test_X',test_X)
    np.save('./work/test_log',test_log)
    np.save('./work/test_Y',test_Y)

def evaluate_automaticThreshold():

    print('Obtain the decision threshold via the precision-recall curve')

    if not os.path.exists('reports_automatic'):
        os.mkdir('reports_automatic')

    val_log  = np.load('./work/val_log.npy')
    val_Y    = np.load('./work/val_Y.npy')

    test_X   = np.load('./work/test_X.npy')
    test_log = np.load('./work/test_log.npy')
    test_Y   = np.load('./work/test_Y.npy')

   
    ## calculate the threshold on valdation set
    ## the predicton probabilites on the validation set
    x_probs = np.copy(val_log)
    x_probs[x_probs==0] = np.nan
    x_probs_min =  np.nanmin(x_probs,axis=1) 
   

    precision, recall, thresholds = precision_recall_curve(val_Y, x_probs_min, pos_label=0)
    thresholds = thresholds 

    f1_score = 2*((precision*recall)/(precision+recall))

    f1_max = np.where(f1_score == np.nanmax(f1_score))[0]

    best_th = thresholds[f1_max]
    print('Best threshold found: ', best_th)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.savefig('./reports_automatic/Precision_Recall_validation.png')

    ## now use the threshold to calculate perform the evaluation on the remaining test set
    print('Evaluate the test set with the automatically obtained threshold')
    y_test = np.copy(test_X)
    y_test = y_test[:,-1,:]
    y_test = np.squeeze(y_test)

    all_predicts = np.copy(y_test)
    idx_th = np.where((test_log < best_th) & (test_log > 0.0) )
    all_predicts[idx_th] = -1 
    n_samples, n_words = np.shape(y_test)

    for h in range(1,4):
            good_idx = []
            bad_idx = []
            thresh = h
                
            for i in tqdm(range(n_samples),ncols=80):
                ## look only at tokens between the start and endtoken
                end_idx = np.where( y_test[i] == 102)[0][0]
                actual_pred = all_predicts[i]
                actual_pred = actual_pred[1:end_idx]
                idx_m = np.where(actual_pred == -1)[0]
                if len(idx_m) < thresh:
                    good_idx.append(i)
                else:
                    bad_idx.append(i)

            good_true = np.where(test_Y == 0)[0]
            bad_true = np.where(test_Y == 1)[0]

            print('Good predicted = ', len(good_idx),'  Good true = ', len(good_true))
            print('Bad predicted = ', len(bad_idx),'  Bad true = ', len(bad_true))


            p_lab = np.zeros(n_samples)
            p_lab[bad_idx] = 1
                
            # print the final results
            report = pd.DataFrame(classification_report(test_Y,p_lab, output_dict=True))
            report.to_csv('reports_automatic/new_class_report_short_prob_h%i.csv'%(h))

def evaluate_fixedThreshold():

    val_X    = np.load('./work/val_X.npy')
    val_log  = np.load('./work/val_log.npy')
    val_Y    = np.load('./work/val_Y.npy')

    test_X   = np.load('./work/test_X.npy')
    test_log = np.load('./work/test_log.npy')
    test_Y   = np.load('./work/test_Y.npy')


    if not os.path.exists('reports_fixed'):
        os.mkdir('reports_fixed')    


    print('Evaluate the validation set on the predefined threshold to obtain the best one.')

    top_g = 5
    thresh_list = np.linspace(1e-3,1e-5,top_g)

    ### first vary th and h on the validation set    

    y_val = np.copy(val_X)
    y_val = y_val[:,-1,:]
    y_val = np.squeeze(y_val)

    n_samples, padd = np.shape(val_log)

    best_th = thresh_list[0]
    best_h = 1
    high_F1 = 0
    
    for t in range(top_g):
        all_predicts = np.copy(y_val)
        ## if the logit score is below the threshold, set it to -1
        idx_th = np.where((val_log < thresh_list[t]) & (val_log > 0.0) )
        all_predicts[idx_th] = -1           
        n_samples, n_words = np.shape(y_val)

        for h in range(1,4):
            good_idx = []
            bad_idx = []
            thresh = h

            for i in range(n_samples):
                ## look only at tokens between the start and endtoken
                end_idx = np.where( y_val[i] == 102)[0][0]
                actual_pred = all_predicts[i]
                actual_pred = actual_pred[1:end_idx]
                idx_m = np.where(actual_pred == -1)[0]
                if len(idx_m) < thresh:
                    good_idx.append(i)
                else:
                    bad_idx.append(i)

            good_true = np.where(val_Y == 0)[0]
            bad_true = np.where(val_Y == 1)[0]

            p_lab = np.zeros(n_samples)
            p_lab[bad_idx] = 1
                
            # save the results on the validation set
            report = pd.DataFrame(classification_report(val_Y,p_lab, output_dict=True))
            report.to_csv('reports_fixed/validation_class_report_short_prob_t%i_h%i.csv'%(t,h)) 
            f1 = f1_score(val_Y,p_lab)
            if f1 > high_F1:
                high_F1 = f1
                best_th = thresh_list[t]
                best_h = h
   

    print('Highest F1: ', high_F1, ' Best th: ', best_th, ' Best h: ', best_h)

    ## now use the threshold to calculate perform the evaluation on the remaining test set
    print('Evalute the test set on the best threshold.')

    y_test = np.copy(test_X)
    y_test = y_test[:,-1,:]
    y_test = np.squeeze(y_test)

    all_predicts = np.copy(y_test)
    idx_th = np.where((test_log < best_th) & (test_log > 0.0) )
    all_predicts[idx_th] = -1 
    n_samples, n_words = np.shape(y_test)

    
    good_idx = []
    bad_idx = []
                
    print('Look for anomalies')

    for i in tqdm(range(n_samples),ncols=80):
        ## look only at tokens between the start and endtoken
        end_idx = np.where( y_test[i] == 102)[0][0]
        actual_pred = all_predicts[i]
        actual_pred = actual_pred[1:end_idx]
        idx_m = np.where(actual_pred == -1)[0]
        if len(idx_m) < best_h:
            good_idx.append(i)
        else:
            bad_idx.append(i)

    good_true = np.where(test_Y == 0)[0]
    bad_true = np.where(test_Y == 1)[0]

    print('Good predicted = ', len(good_idx),'  Good true = ', len(good_true))
    print('Bad predicted = ', len(bad_idx),'  Bad true = ', len(bad_true))


    p_lab = np.zeros(n_samples)
    p_lab[bad_idx] = 1
                
    #print(classification_report(labels,p_lab))
    report = pd.DataFrame(classification_report(test_Y,p_lab, output_dict=True))
    report.to_csv('reports_fixed/Test_Set_class_report_short_prob_h%i.csv'%(best_h))


def create_plots():

    test_X   = np.load('./work/test_X.npy')
    test_log = np.load('./work/test_log.npy')
    test_Y   = np.load('./work/test_Y.npy')


    y_test = np.copy(test_X)
    y_test = y_test[:,-1,:]
    y_test = np.squeeze(y_test)

    n_samples = 5
    start_t = 101
    end_t = 102

    good_true = np.where(test_Y == 0)[0]
    np.random.shuffle(good_true)
    bad_true = np.where(test_Y == 1)[0]
    np.random.shuffle(bad_true)


    good_samples = good_true[0:n_samples]
    
    bad_samples = bad_true[0:n_samples]

    thresh = 5.050e-04

    cmap = plt.get_cmap('RdYlGn')
    norm = MidPointLogNorm(vmin= thresh/5, vmax=1.15, midpoint=thresh)

    plt.figure(figsize=(6,4))
    for n in range(n_samples):
        act_y = y_test[good_samples[n]]
        act_y = act_y[act_y>0]
        act_string = deTokenize(y_test[good_samples[n]], start_t, end_t)
        comp = test_log[good_samples[n]]
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
    plt.colorbar(sm, location='bottom', orientation='horizontal', shrink = 0.75, anchor=(0.5, .9),pad=0.005) 
    plt.savefig('good_strings_uniq',bbox_inches='tight')


    plt.figure(figsize=(6,4))
    for n in range(n_samples):
        act_y = y_test[bad_samples[n]]
        act_y = act_y[act_y>0]
        act_string = deTokenize(y_test[bad_samples[n]], start_t, end_t)
        comp = test_log[bad_samples[n]]
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
    plt.colorbar(sm, location='bottom', orientation='horizontal', shrink = 0.5, anchor=(0.5, .9),pad=0.005)   
    plt.savefig('bad_strings_uniq',bbox_inches='tight')


def main():

    print('Start evaluation')

    if not os.path.exists('work'):
        os.mkdir('work')

    ## create the final validation and test set
    createValidationTest()

    ## obtain the decission threshold automatically and evaluate
    evaluate_automaticThreshold()

    ## evaluate with fixed, predefined thresholds
    evaluate_fixedThreshold()

    ## create some plots
    create_plots()


if __name__ == "__main__":
    main()
