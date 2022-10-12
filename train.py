import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm

import datetime
import os

from cct import create_cct_model
from transformers import BertTokenizer

import custom_loader


#####
# Python Script to train the CCT on the Anomaly data
# The CCT is trained on a taks similar to the Masked masked language task
# See: https://keras.io/examples/nlp/masked_language_modeling/
####



def train():
    print('Train on BGL in a self-supervised fashion a CCT using WordPiece')
    

    save_dir='model' # directory to save the final model
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)



    ## load the original log file
    log_file = "../../data/BGL.log"


    ## Hyperparameters and constans
    positional_emb = True
    conv_layers = 2 # how many convoltional layers should be used
    projection_dim = 64 # dimensionalilty for the transformer encoder
    conv_output_channels = [128,projection_dim]

    num_heads = 5 # number of attention heads
    transformer_units= [projection_dim,projection_dim]    
    transformer_layers = 5 # number of layers in the transformer encoder
    
    learning_rate = 0.0001
    weight_decay = 0.00001
    batch_size = 64
    num_epochs = 100

    windows_wide = 30 # numbers of tokens iin one log line
    windows_size = 15 # number of consectuive log lines in one log window
    step = 1 


    data_dir='data' # data so save the preprocessed data
    ## if the 'data' did not already exists, create it and perform the data preprocessing
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        print('Start preprocessing')
        x_tr, y_tr_window, y_tr_log = custom_loader.load_logfile(
             log_file, windows_size=windows_size, windows_wide = windows_wide,
             step_size=step, NoWordPiece=1)
        

        x_tr = np.asarray(x_tr)
        y_tr_window = np.asarray(y_tr_window)
        y_tr_log = np.asarray(y_tr_log)

        ## shuffle the dataset and create final train and test set
        idx_list = np.linspace(0,len(x_tr)-1,len(x_tr),dtype='int32')
        print(idx_list)
        np.random.shuffle(idx_list)
        x_tr = x_tr[idx_list]
        y_tr_window = y_tr_window[idx_list]
        y_tr_log = y_tr_log[idx_list]


        ## Create train and test set
        ## Labels for the train are set over the labels for the complet window
        ## to sort out every "anamoly" logline late (see below)
        ## Labels for test set are set over the labels only for the last row

        split = 0.6
        n_train = int(len(x_tr)*split)
        x_train = x_tr[0:n_train]
        x_test = x_tr[n_train:]

        y_train = y_tr_window[0:n_train]
        y_test = y_tr_log[n_train:]

        np.save('./data/BGL_masked_Xtrain',x_train)
        np.save('./data/BGL_masked_Xtest',x_test)
        np.save('./data/BGL_masked_Ytrain',y_train)    
        np.save('./data/BGL_masked_Ytest', y_test)
  
    ## load the preprocessed data
    x_train= np.load('./data/BGL_masked_Xtrain.npy') 
    x_test = np.load('./data/BGL_masked_Xtest.npy')
    y_train = np.load('./data/BGL_masked_Ytrain.npy')   
    y_test =  np.load('./data/BGL_masked_Ytest.npy')


    print(np.shape(x_train), np.shape(y_train))


    idx_good = np.where(y_train == 0)[0] # label == 0 -> no anomaly // label==1 -> anomaly
    idx_bad =  np.where(y_train == 1)[0]


    
    n_sampl, n_ws, pad = np.shape(x_train)
    c_train = np.zeros((n_sampl, n_ws))
    mask_train = np.zeros((n_sampl, n_ws, pad))
    for i in range(n_sampl):
        sample = x_train[i]
        idx_s, idx_t = np.where(sample>0)
        mask_train[i, idx_s, idx_t] = 1
        for j in range(n_ws):
            c_train[i,j] = len(np.where(idx_s==j)[0])


    x_train = x_train[idx_good] # train only on the "good" examples
    c_train = c_train[idx_good]
    mask_train = mask_train[idx_good]


    ## create the sample weights
    c_train_last = c_train[:,-1]   

    hist, bin_edges = np.histogram(c_train_last, bins = 1+int(windows_wide-np.min(c_train_last)))    
    hist = hist/np.sum(hist) # divide with the total number

    train_w = np.log(hist+1e-5)/np.log(1/len(hist)) # use the logarithm of 1/2 to calculate the weights
    ## create the sample weights
    list_l = np.linspace(int(np.min(c_train_last)),windows_wide,int(windows_wide-np.min(c_train_last))+1, dtype='int32')
    print(len(train_w), len(list_l))
    sample_w = np.zeros(n_sampl)
    for i in range(len(list_l)):
        idx_sample = np.where(c_train[:,-1] == list_l[i])[0]
        sample_w[idx_sample] = train_w[i]
    

    ## build y_train and y_test again for the masked prediction task    
    ## get the id of the [MASK] token
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    ids = bert_tokenizer(bert_tokenizer.mask_token)
    ids = ids['input_ids']## should be something like [101,103,102] -> 101: Start token / 102: End token
    mask_id = ids[1]
    
    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    x_train = np.expand_dims(x_train,-1)
    x_test = np.expand_dims(x_test, -1)

    mask_train = np.expand_dims(mask_train, -1)


    
    ## additional 'mask' to say, where are the important information
    x_train = np.concatenate((x_train,mask_train),-1)



    x_train = x_train.astype('int32')
    x_test = x_test.astype('int32')

    y_train = np.copy(x_train)
    y_test = np.copy(x_test)

    print(np.shape(np.where(x_train>0)))


    #print(inpt_shape)
    print(np.shape(x_train), np.shape(y_train))
    print(np.shape(x_test), np.shape(y_test))


    inpt_shape = np.shape(x_train)[1:]
    print('Input Shape:', inpt_shape)

    ### create the CCT model
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

    ### Prepare training
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    cct_model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.SparseCategoricalCrossentropy()],)#[tf.keras.metrics.MeanSquaredError()],)

    checkpoint_filepath = "./model/checkpoint.h5"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )

    print(cct_model.summary())

    
    ## additional callback to log, to use it later in tensorboard
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    ## start training
    #history = cct_model.fit( x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0.1, callbacks=[checkpoint_callback,tensorboard_callback]  )


    n_batches = int(len(x_train)/batch_size)

    print(n_batches * batch_size, len(x_train))
    n_val = len(x_train) - (n_batches * batch_size)
    val_train = x_train[-n_val:]
    val_y = y_train[-n_val:]
    x_train = x_train[:-n_val]
    y_train = y_train[:-n_val]
    c_train = c_train[:-n_val]
    sample_w = sample_w[:-n_val]
    e_loss = []
    val_loss = []

    print('Shape of validation and train:')
    print(np.shape(val_train), np.shape(val_y))
    print(np.shape(x_train), np.shape(y_train))



    ##some parameters for the masked token task
    perc_masked =0.2# % of not padded(!) tokens to mask
    idx = np.linspace(0,len(x_train)-1,len(x_train),dtype='int32')

    for e in range(num_epochs):
        print('Epoch %i of %i'%(e+1,num_epochs))
        b_loss = []
        ##shuffle a bit
        np.random.shuffle(idx)
        X = x_train[idx]
        Y = y_train[idx]
        C = c_train[idx]
        W = sample_w[idx]

        for b in tqdm(range(n_batches-1),ascii=True,ncols=80): ##NOTE: -1 to save the last batch for evaluation
            # get the actual batch
            x_batch = X[0+(batch_size*b):batch_size+(batch_size*b)]
            y_batch = Y[0+(batch_size*b):batch_size+(batch_size*b),-1,:,0]
            c_batch = C[0+(batch_size*b):batch_size+(batch_size*b),-1]
            sw_batch = W[0+(batch_size*b):batch_size+(batch_size*b)]
            ## include the ID of the mask token to X/Y contains the "correct" values
            for s in range(len(x_batch)):
                mask_idx = np.random.choice(int(c_batch[s])-1, int(c_batch[s]*perc_masked),replace=False)
                x_batch[s,-1,mask_idx+1] = mask_id # set with mask_id
                #print(mask_idx)
                #print(x_batch[s,-1])
                #print(y_batch[s])
            #print(np.shape(x_batch),np.shape(y_batch) )
            #print('#########')
            history = cct_model.train_on_batch( x_batch, y_batch,sample_weight = sw_batch, reset_metrics=False, return_dict=True )
            b_loss.append(np.mean(history['loss']))
            
        e_loss.append(np.mean(b_loss))
        ## validate on validation set
        x_batch = val_train#val_train[0+(batch_size*n_batches-1):batch_size+(batch_size*n_batches-1)]
        y_batch = val_y[:,-1,:,0]#Y[0+(batch_size*n_batches-1):batch_size+(batch_size*n_batches-1),-1,:,0]

        history = cct_model.test_on_batch(x_batch, y_batch, reset_metrics=False, return_dict=True)
        print(np.mean(b_loss), np.mean(history['loss']))
        val_loss.append(np.mean(history['loss']))

    cct_model.save_weights(save_dir+'/cct_model.hdf5')

    ## plot losses 
    plt.figure()
    plt.plot(e_loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation Losses Over Epochs", fontsize=14)
    plt.legend()
    plt.grid()
    plt.savefig('loss')


if __name__ == "__main__":
    train()
