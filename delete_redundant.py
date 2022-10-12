import numpy as np
from tqdm import tqdm

def compare():

        print('Compare train and test data in matters of identical samples')

        x_train = np.load('./data/BGL_masked_Xtrain.npy')
        x_test  = np.load('./data/BGL_masked_Xtest.npy')
        y_test  = np.load('./data/BGL_masked_Ytest.npy')

        print(np.shape(x_train), np.shape(x_test))

        ## just check about identical samples, labeld as good
        # NOTE: why not bad samples? -> Because Xtrain did not contain any bad samples anymore
        idx_good = np.where(y_test == 0)[0]

        x_test_good = x_test[idx_good]

        x_train = x_train
        x_test_good = x_test_good

        ## remove all redundent elements
        print('Remove all redundent elements')
        ## get the unique elements in the training set
        x_train = np.unique(x_train,axis=0) # NOTE: the numpy.unique call here needs a lot of time !
        print(np.shape(x_train))

        unique_test = []
        unique_labls = []

        n_uniq_sampls = len(x_train)
        n_chunks = 3
        n_sampls_c = n_uniq_sampls//3


        ## calculate sum of train and test for the first heuristic
        sum_train = np.sum(x_train,axis=(1,2))
        sum_test = np.sum(x_test_good,axis=(1,2))

        print('Start')
        ## iterate only over the "good" samples
        for i in tqdm(range(len(x_test_good)),ncols=80):

            ## get only the samples from the training set, which have the same sum
            train_idx = np.where(sum_train == sum_test[i])[0]
            sub_train = np.copy(x_train[train_idx])    

            check =( x_test_good[i] == sub_train)*1 # convert to zero (false) and 1 (true)
            self_check =( x_test_good[i] == x_test_good[i])*1 # convert to zero (false) and 1 (true)
            check_sum = np.sum(check,axis=(1,2)) #  sum over all 1 / trues
            check_sum_self =np.sum(self_check)
            idx = np.where(check_sum == check_sum_self)[0]
            
            if len(idx) == 0: # is an unique sample
                # find all samples of this unique sample in the original dataset
                unique_test.append(x_test_good[i])
                unique_labls.append(0)
           




        print(np.shape(unique_test), np.shape(unique_labls))
        ## append the "bad" samples
        idx_bad = np.where(y_test == 1)[0]
        for i in tqdm(range(len(idx_bad)),ncols=80):
            unique_test.append(x_test[idx_bad[i]])
            unique_labls.append(1)

        unique_test = np.asarray(unique_test)
        unique_labls = np.asarray(unique_labls)

        print(np.shape(unique_test), np.shape(unique_labls))

        ###shuffle
        idx = np.linspace(0,len(unique_labls)-1,len(unique_labls),dtype='int32')
        np.random.shuffle(idx)
        unique_test = unique_test[idx]
        unique_labls = unique_labls[idx]

        print('Save the new test samples')
        np.save('./data/BGL_masked_Xtest_uniq.npy',unique_test)
        np.save('./data/BGL_masked_Ytest_uniq.npy',unique_labls)



if __name__=="__main__":
    compare()
