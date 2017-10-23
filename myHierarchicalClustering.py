import numpy as np
from scipy.spatial import distance as Distance
from sklearn import datasets

global final_clusters

def hierarchical_clustering(data,threshold,linkage='single'):
    #step 1: Calculate and store distances between all points
        distance_Array = getDistanceArray( data )
    #Step 2: Initialize dictionary with individual elements as singular clusters
        old_clusters={}
        for i in range(data.shape[0]):
            old_clusters[i]=[i]
        #print "Inital clusters:",old_clusters

    #Step 3:

        # clusters where data points are represented by their index in the original data
        myClusters=clusterize(old_clusters,distance_Array,threshold,linkage)

        #Put co-ordinates in place of the indices of the data points
        for cluster_id in range(len(myClusters)):
            for elem in range(len(myClusters[cluster_id])):
                index_in_data=myClusters[cluster_id][elem]
                myClusters[cluster_id][elem]=data[index_in_data]
        return {
            'myClusters':myClusters,
            'noClusters':len(myClusters)
        }

def clusterize(old_clusters, distance_Array,threshold,linkage):
    index, distance = getMinDistanceClusters( old_clusters, distance_Array,linkage=linkage)

    if (not isinstance( threshold, int )) and (len(old_clusters)==1):
        return old_clusters
    if isinstance( threshold, int ) and distance>threshold:
        return old_clusters
    row,column=index
    new_clusters = {}
    for i in range( len( old_clusters ) - 1 ):
        new_clusters[i] = None
    #print "Minimum dist found between clusters=", row,column
    print "Min Dist=",distance
    # This is for single linkage
    if linkage=='single':
        distance_Array[index[0]][index[1]] = np.inf
        distance_Array[index[1]][index[0]] = np.inf
    elif linkage=='complete':
        print "complete"
        distance_Array[index[0]][index[1]] = -np.inf
        distance_Array[index[1]][index[0]] = -np.inf
    # for complete linkage final=== {0: [0, 3, 6], 1: [1, 4], 2: [2, 7, 8], 3: [5], 4: [9]}
    j=0
    found=False
    for i in range(len(old_clusters)):
        current_cluster=old_clusters[i]
        if row in current_cluster or column in current_cluster:
            if found:
                for elem in current_cluster:
                    new_clusters[same_cluster_index].append(elem)
            else:
                found = True
                same_cluster_index=j
                if new_clusters[j]==None:
                    new_clusters[j]= current_cluster
                else:
                    for elem in current_cluster:
                        new_clusters[j].append(elem)
                j+=1
        else:
            if new_clusters[j]==None:
                new_clusters[j]= current_cluster
            else:
                for elem in current_cluster:
                    new_clusters[j].append(elem)
            j+=1

    #print "i==",i
    print"new clusters=",(new_clusters)
    #final_clusters=new_clusters.copy()
    return clusterize(new_clusters,distance_Array,threshold,linkage=linkage)

def getMinDistanceClusters(clusters,distance_Array,linkage='single'):
    minDist = np.inf
    minEle = [[-1, -1],minDist]
    for cluster_id1,cluster1 in clusters.items():
        for cluster_id2, cluster2 in clusters.items():
            if(cluster_id1!=cluster_id2):
                index,dist=getLinkageDistance( cluster1, cluster2 , distance_Array,linkage=linkage)
                if(minDist>dist):
                    minDist=dist
                    minEle=[index,dist]
    return minEle[0],minEle[1]


def getLinkageDistance(cluster1,cluster2,distArray,linkage='single'):
    array = [[i, j, distArray[i][j]] for i in cluster1 for j in cluster2]
    if(linkage=="single"):
        minDist=-1
        #for i in cluster1:
            #return min([getEuclideanDistance(i,j) for i in cluster1 for j in cluster2])
        distElement=min(array,key= lambda x:x[2])
        return distElement[:2],distElement[2] #return index of the elements in the distance array and the distance
    if(linkage=='complete'):
        distElement=max(array,key= lambda x:x[2])
        #print "max dist==",distElement[2]
        return distElement[:2], distElement[2]  # return index of the elements in the distance array and the distance

def getMinDisCluster(distanceArray):
    shape=distanceArray.shape
    index=distanceArray.ravel().argmin()
    return np.unravel_index(index,shape),distanceArray.ravel()[index]


def getDistanceArray(data):
    distance_Array = []
    for i in range( data.shape[0] ):
        for j in range( data.shape[0] ):
            if (i == j):
                distance_Array.append( np.inf )
            else:
                # distance_Array.append(
                #     getLinkageDistance( data[i][np.newaxis], data[j][np.newaxis], distance_Array,linkage='single' ) )
                #
                distance_Array.append(Distance.euclidean(data[i],data[j]))
    #print "dist array==",np.array( distance_Array ).reshape( data.shape[0], data.shape[0] )
    return np.array( distance_Array ).reshape( data.shape[0], data.shape[0] )

def readFileInArray(filePath,delimeter,inType,skip_header=0,skip_col=0):
    X = np.genfromtxt(str(filePath), dtype=str(inType), skip_header=int(skip_header),delimiter=str(delimeter))
    if skip_col!=0:
        return X[:,skip_col:]
    else:
        return X

#data from file



# result=hierarchical_clustering(data,0.2,linkage='single')
# print "\nNo of Clusters formed:\t",result['noClusters']
# print "\nfinal clusters:\n",result['myClusters']

#distance_Array = getDistanceArray( data1 )
# myclusters={}
# for i in range(data1.shape[0]):
#     myclusters[i]=[i]
# print "clusters:",myclusters
# print getMinDistanceClusters( myclusters, distance_Array )
#print getLinkageDistance(myclusters[0],myclusters[1],distance_Array,linkage="single")