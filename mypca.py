# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 15:53:19 2017

@author: ptambvek
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as lalg
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.mlab import PCA
from statsmodels.multivariate.pca import pca

def readFileInMatrix(filePath,delimeter,inType):
    X = np.genfromtxt(str(filePath), dtype=str(inType), skip_header=1,delimiter=str(delimeter))
    return X

def myPCA(x,standardize):
    means = x.mean(axis=0) #find mean along columns
    if(standardize == True):
        x_mean_centered = x-means #mean center the data
    else:
        x_mean_centered = x
    # plot of PCA scores
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1,projection='3d')
    ax.set_title('Original data plot')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter(x[:, 0],x[:, 1],x[:,2] ,color='blue')
    fig.show()

    #find co-variance matrix
    # *********important************* cov should have rowvar=false when variables are along columns and data along rows
    x_cov = np.cov(x_mean_centered,rowvar=False)

    #find eigenvalues and eigenvectors of the covariance matrix
    eig_val,eig_vect= np.linalg.eig(x_cov)

    #sort eigenvalues in descending order::

    # step 1. find indices that would sort the eigen values.
    sort_indices=eig_val.argsort()
    # step 2. reverse the indices so that it will give indeces for descending sorted array
    sort_indices=sort_indices[::-1]
    # step 3. arraynge eigen values according to these indeces.
    eig_val = eig_val[sort_indices]
    # step 4. arrange eigen vectors according to the indices of corresponding eigen values.
    eig_vect = eig_vect[:,sort_indices]
    #get PCA Scores by multiplying the mean centered data with the eigenvectors.

    pcaScores= x_mean_centered.dot(eig_vect)

    PCAObj = {
        'x_mean_centered': x_mean_centered,
        'eig_vectors':eig_vect,
        'lambdas':eig_val,
        'pca_Scores':pcaScores
    }
    return PCAObj

# routine to calculate percent variance for given PC
def getPercentVariance(pcaObject,pcNumber):
    eigenValues=pcaObject['lambdas']
    numerator = sum(eigenValues[range(0,pcNumber)])
    denominator = sum(eigenValues)
    percentVariance= 100* (numerator/denominator)
    return round(percentVariance,2)

def getOriginalXMeanCentered(pcaResult):
    Y=pcaResult['pca_Scores']
    print np.shape(Y)
    P=pcaResult['eig_vectors'][0]
    p_inv= P.T
    orig_mean_centered= Y.dot(p_inv)
    return orig_mean_centered



# def main():
#     X = np.genfromtxt('E:\UNCC\Acad\Machine Learning\Quizz\Quiz1\dataset_Quizz.csv', dtype='float', skip_header=1,delimiter=',')
#     pcaResult=myPCA(X)
#     print(pcaResult['pca_Scores'])
#
#     pc1= getPercentVariance(pcaResult,1)
#     print "percent variance for pc1= " + str(pc1) + "%"
#
#     pc2 = getPercentVariance(pcaResult, 2)
#     print "percent variance for pc2= " + str(pc2) + "%"
#
#
#     #plot of PCA scores
#     fig = plt.figure()
#     ax= fig.add_subplot(1,1,1,projection='3d')
#     ax.set_title('Scores plot')
#     ax.set_xlabel('PC1')
#     ax.set_ylabel('PC2')
#     ax.set_zlabel('PC3')
#     ax.scatter(pcaResult['pca_Scores'][:,0],pcaResult['pca_Scores'][:,1],color='blue')
#     fig.show()
#
#     #scree plot: ie plot of eigen values vs componnent number
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     ax.set_title('Scree plot')
#     ax.scatter(range(len(pcaResult['lambdas'])),pcaResult['lambdas'], color='blue')
#     fig.show()
#
#     # loadings plot:  plot of PC1 vs PC2
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     ax.set_title('loadings plot')
#     ax.scatter(pcaResult['eig_vectors'][:,0],pcaResult['eig_vectors'][:,1] ,color='blue')
#     fig.show()
#
#
# if __name__=='__main__':
#     main()
