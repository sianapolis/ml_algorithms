import numpy as np

class LinearClassifiers:
    def __init__(self,nclass):
        self.nclass = nclass

    @staticmethod
    def ols_weights(X,y):
        X = np.hstack((X, np.ones((X.shape[0],1))))
        x_t = np.transpose(X)
        a = x_t @ X
        b = x_t @ y

        weights = np.linalg.inv(a) @ b
        return weights
    
    @staticmethod
    def separate_classes(full_Y,full_X, kclass):
        Y_1 = (full_Y == kclass).nonzero()[0]
        Y_2 = (full_Y != kclass).nonzero()[0]

        X_1 = full_X[Y_1,:]
        X_2 = full_X[Y_2,:]

        return X_1,X_2
    
    @staticmethod
    def softmax(Y):
        normalisation_part = np.sum(np.exp(Y))
  
        softmax_ret = np.exp(Y)/normalisation_part

        return softmax_ret
    
    def prob_class(self,A,B,test):
        examplesA_Y = np.full(A.shape[0],0)
        examplesB_Y = np.full(B.shape[0],1)

        X = np.append(A,B,axis=0)
        Y = np.append(examplesA_Y,examplesB_Y,axis=0)

        w = self.ols_weights(X,Y)
        
        x = np.hstack((test,np.ones((test.shape[0],1))))
        x_t = np.transpose(x)
        W = np.transpose(w)
        preds = W @ x_t

        preds = np.round(preds,0)
        return preds

    
    def multiclass_linear(self,X_train,Y_train,test):
        if self.nclass <= 2:
            A,B = self.separate_classes(Y_train,X_train,0)
            predicted = self.prob_class(A,B,test)
        else:
            predicted_array = np.array([])

            for i in np.unique(Y_train):
                A, B = self.separate_classes(Y_train,X_train, i)
                pred_class = self.prob_class(A,B,test)

                predicted_array = np.append(predicted_array,pred_class,axis=0)

            predicted_array = np.reshape(predicted_array,(len(np.unique(Y_train)),test.shape[0]))

            for abba in range(predicted_array.shape[1]):
                predicted_array[:,abba] = self.softmax(predicted_array[:,abba])
            predicted = predicted_array.T
            predicted = np.argmax(predicted,axis=1)



        return predicted