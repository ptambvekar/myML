
## Pranav Tambvekar ##

import numpy as np
import  numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets


def doLogistic(X,Y,iterations,theta):
    alpha = 0.5
    for i in range(iterations):
        theta = doGradientDescent(X,Y,theta,alpha)
    return theta

def doGradientDescent(X,Y,theta,alpha):
    theta_new = []
    for j in range( len( theta ) ):
        derivative = partial_derivative( X, Y, theta, j,alpha )
        theta_val = theta[j] - (derivative)
        theta_new.append( theta_val )   # update the theta in each iteration
    return theta_new

def partial_derivative(X,Y,theta,index_theta,alpha):
    pd = 0
    m=X.shape[0]
    for index_sample in range(m):
        h_theta = getOutput(theta,X[index_sample])
        residual= (h_theta-Y[index_sample])
        pd_intermediate = residual*X[index_sample][index_theta]
        pd = pd+pd_intermediate
    return (alpha/m)*pd


def getOutput(theta, X):
    z_current = 0
    for i in range(len(theta)):
        z_current=z_current + (X[i] * theta[i]) #compute the linear oombination of theta and X
    return sigmoid(z_current)

def predict_new(X, logisticResults):

    if getOutput(logisticResults,X) >= 0.5:
        return 1
    else:
        return 0


# sigmoid function
def sigmoid (x):
    return float(1.0)/float((1.0 + np.exp(-x)))




#####################################################

def main():

    trials=100
    irisData=datasets.load_iris()
    Y=irisData.target


    X=irisData.data
    # Take only petal data
    X = irisData.data[:, 2:]

    #removing setosa from target and input data:
    setosa_indices=np.where(Y==0)
    Y= Y[50:]
    Y[np.where( Y == 1 )] = 0
    Y[np.where( Y == 2 )] = 1
    Y=Y.reshape(100,1)
    X= X[50:,:]

    # Scale using the given formula
    X[:,0] = (X[:,0] - np.min(X[:,0])) / (np.max(X[:,0]) - np.min(X[:,0]))
    X[:, 1]= (X[:, 1] - np.min( X[:, 1] )) / (np.max( X[:, 1] ) - np.min( X[:, 1] ))

    # adding an extra column all 1, for calculation of bias (beta_0)
    ones=np.ones((X.shape[0],1))
    X=np.concatenate((ones,X),axis=1)

    error=0
    for i in range(trials):
        X_training=X.copy()
        Y_training=Y.copy()
        index=i
        X_test = X_training[index]  # Chose ith observation as test data and the remaining as the training data
        Y_test = Y_training[index]
        X_training=np.array((np.delete(X_training,index,axis=0)))
        Y_training = np.delete( Y_training, index,axis=0 )

        theta = [1] * (X_training.shape[1])

        result=doLogistic( X, Y, 5000,theta )
        Y_calc= predict_new(X_test,result) #test the trained model for the test data

        print "actual==", Y_test
        print "calculated==", Y_calc

        if not np.equal(Y_test,Y_calc):
            error=error+1;
        print "Iteration: ", i
    print "Error is:",error," for 100 trials"


if __name__=='__main__':
    main()