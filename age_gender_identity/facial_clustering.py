from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import cv2
import time

from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import metrics

from facial_analysis import FacialImageProcessing,is_image

rankorder_clustering=1
scipy_clustering=2
sklearn_clustering=3
use_clustering=scipy_clustering

if use_clustering==rankorder_clustering:
    import networkx as nx   
    class Neighbour:
        def __init__(self, entity, distance):
            self.entity = entity
            self.distance = distance

    class Face:
        def __init__(self, index = None,absolute_distance_neighbours = None, rank_order_neighbours = None):
            self.index = index
            self.absolute_distance_neighbours = absolute_distance_neighbours
            self.rank_order_neighbours = rank_order_neighbours

    class Cluster:
        def __init__(self):
            self.faces = list()
            self.absolute_distance_neighbours = None
            self.rank_order_neighbours = None
            self.normalized_distance = None
            
    def initial_cluster_creation(faces):
        clusters = []
        for face in faces:
            cluster = Cluster() 
            cluster.faces.append(face)
            clusters.append(cluster)
        return(clusters)
            
    # Create nearest neighbours list of absolute distance
    def assign_absolute_distance_neighbours_for_faces(faces, dist_matrix, N = 20):
        for i, face1 in enumerate(faces):
            nearest_neighbour = []
            print("Calculating neighbours for face {}/{}".format(i + 1, len(faces)), end = "\r")
            for j, face2 in enumerate(faces):
                distance = dist_matrix[i][j]
                neighbour = Neighbour(face2, distance)
                nearest_neighbour.append(neighbour)
            nearest_neighbour.sort(key = lambda x: x.distance)
            face1.absolute_distance_neighbours = nearest_neighbour[0:N]

    def find_nearest_distance_between_clusters(cluster1, cluster2, dist_matrix):
        nearest_distance = sys.float_info.max
        for face1 in cluster1.faces:
            for face2 in cluster2.faces:
                distance = dist_matrix[face1.index][face2.index]
                
                if distance < nearest_distance: 
                    nearest_distance = distance
                    
                # If there is a distance of 0 then there is no need to continue
                if distance == 0:
                    return(0)
        return(nearest_distance)
                

    def find_normalized_distance_between_clusters(cluster1, cluster2, dist_matrix,K = 12):
        all_faces_in_clusters = cluster1.faces + cluster2.faces
        normalized_distance = 0

        for face in all_faces_in_clusters:
            total_absolute_distance_for_top_K_neighbours = sum([neighbour.distance for neighbour in face.absolute_distance_neighbours[0:K]]) 
            normalized_distance += total_absolute_distance_for_top_K_neighbours
        
        # Now average the distance
        K = min(len(face.absolute_distance_neighbours), K)
        normalized_distance = normalized_distance/K
        
        # then divide by all the faces in the cluster
        normalized_distance = normalized_distance/len(all_faces_in_clusters)
        normalized_distance = (1/normalized_distance) * find_nearest_distance_between_clusters(cluster1, cluster2,dist_matrix)
        return(normalized_distance)

        
    def assign_absolute_distance_neighbours_for_clusters(clusters, dist_matrix,K = 20):
        for i, cluster1 in enumerate(clusters):
            nearest_neighbours = []
            for j, cluster2 in enumerate(clusters):
                distance = find_nearest_distance_between_clusters(cluster1, cluster2,dist_matrix)
                neighbour = Neighbour(cluster2, distance)
                nearest_neighbours.append(neighbour)
            nearest_neighbours.sort(key = lambda x: x.distance)
            cluster1.absolute_distance_neighbours = nearest_neighbours[0:K]
            

    def find_asym_rank_order(entity1, entity2):
        penalty = 0
        for i, neighbour1 in enumerate(entity1.absolute_distance_neighbours):
            for j, neighbour2 in enumerate(entity2.absolute_distance_neighbours):
                if neighbour1.entity == neighbour2.entity:
                    if j == 0: # this means that we found the rank of entity2 in entity1's neighbouts
                        return(penalty, i + 1)
                    else:
                        penalty += j
        return(penalty, i+1)

    def find_rank_order(entity1, entity2):
        distance_entity1_entity2, num_neighbours_entity1 = find_asym_rank_order(entity1, entity2)
        distance_entity2_entity1, num_neighbours_entity2 = find_asym_rank_order(entity2, entity1)
        min_neighbours = min(num_neighbours_entity1, num_neighbours_entity2)
        return((distance_entity1_entity2 + distance_entity2_entity1)/min_neighbours)

    def assign_rank_order(entities):
        for entity1 in entities:
            nearest_neighbours = []
            for entity2 in entities:
                rank_order = find_rank_order(entity1, entity2)
                nearest_neighbours.append(Neighbour(entity2, rank_order))

            nearest_neighbours.sort(key = lambda x : x.distance)
            entity1.rank_order_neighbours = nearest_neighbours


    def find_clusters(faces,dist_matrix):
        clusters = initial_cluster_creation(faces)
        assign_absolute_distance_neighbours_for_clusters(clusters,dist_matrix)
        t = 14 #14
        norm_threshold=0.9 #0.98 #0.6 #0.82
        prev_cluster_number = len(clusters)
        num_created_clusters = prev_cluster_number
        is_initialized = False

        while (not is_initialized) or (num_created_clusters):
            print("Number of clusters in this iteration: {}".format(len(clusters)))
            G = nx.Graph()
            for cluster in clusters:
                G.add_node(cluster)
            num_pairs = sum(range(len(clusters) + 1))
            new_cluster_indices=[i for i in range(len(clusters))]
            processed_pairs = 0
            
        #     Find the candidate merging pairs
            for i, cluster1 in enumerate(clusters):
                # Order does not matter of the clusters since rank_order_distance and normalized_distance is symmetric
                # so we can get away with only calculating half of the required pairs
                for cluster_neighbour in cluster1.absolute_distance_neighbours:
                    cluster2 = cluster_neighbour.entity
                    processed_pairs += 1
                    print("Processed {}/{} pairs".format(processed_pairs, num_pairs), end="\r")
                    # No need to merge with yourself 
                    if cluster1 is cluster2:
                        continue
                    else: 
                        normalized_distance = find_normalized_distance_between_clusters(cluster1, cluster2,dist_matrix)
                        #normalized_distance = find_nearest_distance_between_clusters(cluster1, cluster2,dist_matrix)
                        
                        if (normalized_distance >= norm_threshold):
                            continue
                        rank_order_distance = find_rank_order(cluster1, cluster2)
                        if (rank_order_distance >= t):
                            continue
                        G.add_edge(cluster1, cluster2)
            print()     
            clusters = []
            for _clusters in nx.connected_components(G):
                new_cluster = Cluster()
                for cluster in _clusters:
                    for face in cluster.faces:
                        new_cluster.faces.append(face)
                clusters.append(new_cluster)


            current_cluster_number = len(clusters)
            num_created_clusters = prev_cluster_number - current_cluster_number
            prev_cluster_number = current_cluster_number

            assign_absolute_distance_neighbours_for_clusters(clusters,dist_matrix)
            is_initialized = True
            #break

        # Now that the clusters have been created, separate them into clusters that have one face
        # and clusters that have more than one face
        unmatched_clusters = []
        matched_clusters = []

        for cluster in clusters:
            if len(cluster.faces) == 1:
                unmatched_clusters.append(cluster)
            else:
                matched_clusters.append(cluster)
                
        matched_clusters.sort(key = lambda x: len(x.faces), reverse = True)
                
        return(matched_clusters, unmatched_clusters)
elif use_clustering==scipy_clustering:
    import scipy.cluster.hierarchy as hac
    from scipy.spatial.distance import squareform
    clusteringMethod='single'
else:
    from sklearn.cluster import DBSCAN,MeanShift, estimate_bandwidth,AffinityPropagation
    from sklearn.metrics.pairwise import pairwise_distances


def get_facial_clusters(dist_matrix,distanceThreshold=1,all_indices=None,no_images_in_cluster=1):    
    '''
    Perform real clustering

    :param real[][] dist_matrix: The matrix of pair-wise distances between facial feature vectors
    :param real distanceThreshold (optional): The maximum distance between elements in a cluster
    :param int[] all_indices (optional): The list of photo indices of each detected face. It is used to prevent the union into one cluster of two different persons presented on the same photo.
    :param int no_images_in_cluster (optional): The minimum number of images to form a cluster
    :return: the list of clusters. Each cluster is represented by a list of element indices
    :rtype: list
    '''
    clusters=[]
    num_faces=dist_matrix.shape[0]
    #print('num_faces:',num_faces)

    if use_clustering==rankorder_clustering:
        faces = []
        for i in range(num_faces):
            faces.append(Face(index=i))
        assign_absolute_distance_neighbours_for_faces(faces,dist_matrix)
        matched_clusters, unmatched_clusters = find_clusters(faces,dist_matrix)
        print('matched_len:',len(matched_clusters),'unmatched_len:',len(unmatched_clusters))
        #for cluster in matched_clusters:
        #    print([f.index for f in cluster.faces]) 
        clusters=[[f.index for f in cluster.faces] for cluster in matched_clusters]
    elif use_clustering==scipy_clustering:
        condensed_dist_matrix=squareform(dist_matrix,checks=False)
        z = hac.linkage(condensed_dist_matrix, method=clusteringMethod)
        labels = hac.fcluster(z, distanceThreshold, 'distance')
        
        if all_indices is None:
            clusters=[[ind for ind,label in enumerate(labels) if label==lbl] for lbl in set(labels)]
        else:
            for lbl in set(labels):
                cluster=[ind for ind,label in enumerate(labels) if label==lbl]
                if len(cluster)>1:
                    inf_dist=100
                    dist_matrix_cluster=dist_matrix[cluster][:,cluster]
                    penalties=np.array([[inf_dist*(all_indices[i]==all_indices[j] and i!=j) for j in cluster] for i in cluster])
                    dist_matrix_cluster+=penalties
                    condensed_dist_matrix=squareform(dist_matrix_cluster)
                    z = hac.linkage(condensed_dist_matrix, method='complete')
                    labels_cluster = hac.fcluster(z, inf_dist/2, 'distance')
                    clusters.extend([[cluster[ind] for ind,label in enumerate(labels_cluster) if label==l] for l in set(labels_cluster)])
                else:
                    clusters.append(cluster)
    else:
        db = DBSCAN(eps=distanceThreshold, min_samples=no_images_in_cluster,metric="precomputed").fit(dist_matrix) #0.78
        #db=AffinityPropagation().fit(all_features)
        #db = MeanShift(bandwidth=0.7).fit(all_features)
        labels = db.labels_
        clusters=[[ind for ind,label in enumerate(labels) if label==lbl] for lbl in set(labels) if lbl!=-1]
        #cluster_min_dists=[[min([dist_matrix[i,elem] for elem in cluster1 for i in cluster2]) for cluster1 in clusters] for cluster2 in clusters]
        #print('cluster_min_dists:',cluster_min_dists) 
        
        #extend clusters
        if False and all_indices is not None and len(clusters)>0:
            elems_out_of_clusters=set(range(len(all_indices)))-set([elem for cluster in clusters for elem in cluster])
            no_added_images=0
            for i in elems_out_of_clusters:
                min_dists=[min([pair_dist[i][elem][0] for elem in cluster]) for cluster in clusters]
                #min_percentiles=np.array([np.percentile(d,50) for d in dists])
                closest_cluster=np.argsort(min_dists)[0]
                if min_dists[closest_cluster]<distanceThreshold:
                    clusters[closest_cluster].append(i)
                    no_added_images+=1

            print('no of other faces:',len(elems_out_of_clusters),' added:',no_added_images)
    
    
    clusters.sort(key=len, reverse=True)
    return clusters
