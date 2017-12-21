import numpy as np
import matplotlib.pyplot as plt


def singleHiddenNeural(X,Y,hiddenNo,outputNo,iterations):

    #number of features
    s1= X.shape[1]
    #number of neurons in hidden layer
    s2=hiddenNo
    #number of neurons in input layer
    s3=outputNo

    alpha= 0.3
    ##### initialize the bias and weight matrix with random numbers #####
    theta_1= np.random.uniform(size=(s1,s2))    # input to first layer
    theta_2 = np.random.uniform( size=(s2, s3) )    # input to second layer
    b_1= np.random.uniform(size=(1,s2)) # bias for 1st layer that goes to the neurons in hidden layer
    b_2 = np.random.uniform(size=(1,s3)) # bias for the 2nd (i.e. hidden) layer that goes to the nerons in output layer

    total_cost=[] # maintain an array of total costs in each iteration

    for i in range(iterations):

        ##### Forward Porpogation #####
        z_2=  b_1+np.dot(X,theta_1)     # total input to 2nd layer (hidden layer)
        a_2=  activation(z_2)           # output of hidden layer

        z_3= b_2 + np.dot(a_2,theta_2)  # input for 3rd layer (output layer)

        a_3= activation(z_3)            # output of third layer (final output)

        ##### Back Porpogation #####

        Err_a3= Y-a_3                   # Error at output layer
        total_cost.append(-1*np.sum((Y*np.log(a_3) + (1-Y)*np.log(1-a_3)),axis=0)/X.shape[0])
        # gradient at output layer
        derivative_a3= activation_derivative(a_3)
        # change factor at output layer
        change_a3= Err_a3 * derivative_a3

        Err_a2= np.dot(change_a3,theta_2.T) # Error at hidden later is dot product of change factor and theta (weight of connections between hidden and output layer)
        # gradient at hidden layer
        derivative_a2 = activation_derivative( a_2 )
        # change factor at hidden layer
        change_a2=  Err_a2 * derivative_a2

        #update weights and bias
        theta_2 = theta_2 + alpha*np.dot(a_2.T, change_a3)
        theta_1 = theta_1 + alpha*np.dot( X.T, change_a2 )
        b_1 = b_1 + alpha * sum(change_a2)
        b_2 = b_2 + alpha * sum(change_a3)

    return {
        "iterationwise_cost":total_cost,
        "bias_1":b_1,
        "bias_2": b_2,
        'theta_1':theta_1,
        'theta_2': theta_2,
        "training_output":a_3,
        'iterationwise_cost':total_cost
    }

# activation function
def activation (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def activation_derivative(x):
    return x * (1 - x)

def predict_new(X,neuralResults):

    z_2 = neuralResults['bias_1'] + np.dot( X, neuralResults['theta_1'] )  # total input to 2nd layer (hidden layer)
    a_2 = activation( z_2 )  # output of hidden layer

    z_3 = neuralResults['bias_2'] + np.dot( a_2, neuralResults['theta_2'] )  # input for 3rd layer (output layer)

    a_3 = activation( z_3 )  # output of third layer (final output)

    a_3= np.where(a_3>0.5,np.ones([1],dtype=np.int),np.zeros([1],dtype=np.int)) # convert the output to 1 or 0 based on value greater than 0.5

    return a_3
