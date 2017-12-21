
import numpy as np
import scipy.stats as stats

beta0_hat=0;
beta1_hat=0

def doLinearReg(x,y):
    global beta1_hat
    global beta0_hat
    x_bar=np.mean(x)
    y_bar=np.mean(y)
    n=x.shape[0]
    beta1_hat= ( np.sum(x*y) - n * x_bar * y_bar ) / ( np.sum(x**2) - (n * (x_bar**2)))
    beta0_hat= y_bar - beta1_hat*x_bar

    y_hat= beta0_hat + beta1_hat*x

    return{
        'y_hat':y_hat
    }


def testModel(X):
    global beta0_hat
    global beta1_hat
    y_test = beta0_hat + beta1_hat * X
    return y_test
