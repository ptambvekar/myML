
import numpy as np
import random
from scipy.spatial import distance
import pandas as pd


def myK_Means(noClusters,X,random):

    # Step 1: choose <noClusters> centroids randomly,
    #initial_Centroids=initialize_centroids(X,noClusters)

    #for quiz, initial centroids= first 3 points
    if(random==False):
        initial_Centroids=X[:noClusters]
    else:
        initial_Centroids=initialize_centroids(X,noClusters)

    old_centroids=initial_Centroids.copy()
    print "\nInitial Centroids:\n",initial_Centroids
    new_centroids=[]
    clusters=np.array([])
    count=0

    # Run loop till converegence
    while True:
        count += 1
        #print "\niteration:\t",count
        # Step 2: calculate dist of each point from the each centroid and assign that point to the nearest (cluster) centroid
        clusters = clusterize( X, old_centroids )
        # Step 3: update centroids to be mean of all the data points in the newly created cluster
        new_centroids = update_centroids( clusters['dictionary'], old_centroids )
        print "\niteration: ",count
        print "\nNew Centroids: ", new_centroids
        if checkConvergence(old_centroids,new_centroids):# exit when the centroids no longer change
            break
        if (count == 1000):   #this condition is for avoiding the infinite loop in case of a large number of clusters
            break
        else:
            old_centroids = new_centroids.copy()
    print "\niterations required:\t",count
    return {
        'clusters_dictionary':clusters['dictionary'],
        'clusters_array':clusters['array'],
        'centroids':new_centroids
    }

#this function checks the equality of new and old centroids
def checkConvergence(old_centroids,new_centroids):
    return np.array_equiv(old_centroids,new_centroids)

#this function finds mean of points in each cluster and returns an array of these means as new centroids.
def update_centroids(clusters,old_means):
    new_means=old_means.copy()
    for cluster_id,cluster_points in clusters.items():
        if(len(cluster_points) < 1): #if a cluster is not assigned any point, then centroid for that cluster will be kept as it was previously.
            new_means[cluster_id]=old_means[cluster_id]
        else:
            new_means[cluster_id]=np.mean(cluster_points,axis=0)
    return new_means

#this function returns object that contains detail cluster-assignment of each point. the same information is returned in different data structures for future use.
def clusterize(X,centroids):
    dataArray=[]
    clusters={}
    k=len(centroids)
    for i in range(k):
        clusters[i]=[]
    for row in X:
        temp = []
        cluster_id=0 #assign cluster id =0 for this point initially
        for i in range(len(centroids)): #create a list of distances of current point under consideration (row) from each mean
            temp.append(distance.euclidean(row,centroids[i]))
        #index of the minimum distance element will be the cluster_id.
        cluster_id= temp.index(min(temp))
        row = np.array( row )[np.newaxis]
        if(len(clusters[cluster_id])>0):
            clusters[cluster_id] = np.append( clusters[cluster_id], row,axis=0 )
        else:
            clusters[cluster_id] = row
        dataArray.append(cluster_id)

    return {
        'dictionary':clusters,
        'array':dataArray,
    }

def initialize_centroids(input_data, k):
    data = input_data.copy()
    np.random.shuffle(data)
    return data[:k]

def getWCSS(centroids,clusters):

    sum=[]
    for i in range(len(centroids)):
        temp = 0
        for j in clusters[i]:
            # print "j==",j
            # print "centroid==",centroids[i]
            temp=temp+ (distance.euclidean(j,centroids[i])**2)
        sum.append(temp)
    return min(sum)

def readFileInArray(filePath,delimeter,inType,skip_header=0,skip_col=0):
    X = np.genfromtxt(str(filePath), dtype=str(inType), skip_header=int(skip_header),delimiter=str(delimeter))
    print X,"\n after skip"
    if skip_col!=0:
        return X[:,skip_col:]
    else:
        return X

# def getIntraDissimilarity(myClusters,centroids):
#     a=[]
#     b=[]
#     for i in range(len(myClusters)):
#        current_cluster=myClusters[i]
#        temp=0
#        for j in current_cluster:
#            current_centroid=centroids[i]
#            temp=temp+distance.euclidean( j, current_centroid )
#            temp=temp/len(current_cluster)
#        #temp is a(i), create array of all a(i)
#        a.append(temp)
#        btemp=sum(a)/len(myClusters)
#        b.append(btemp)


