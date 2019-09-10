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

sys.path.append("..")
from facerec_test import TensorFlowInference,is_image

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
            #print("Calculating neighbours for face {}/{}".format(i + 1, len(faces)), end = "\r")
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
        if normalized_distance!=0:
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


    def find_clusters(faces,dist_matrix,norm_dist_threshold=0.9,rank_threshold = 14):
        clusters = initial_cluster_creation(faces)
        assign_absolute_distance_neighbours_for_clusters(clusters,dist_matrix)
        prev_cluster_number = len(clusters)
        num_created_clusters = prev_cluster_number
        is_initialized = False

        while (not is_initialized) or (num_created_clusters):
            #print("Number of clusters in this iteration: {}".format(len(clusters)))
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
                    #print("Processed {}/{} pairs".format(processed_pairs, num_pairs), end="\r")
                    # No need to merge with yourself 
                    if cluster1 is cluster2:
                        continue
                    else: 
                        normalized_distance = find_normalized_distance_between_clusters(cluster1, cluster2,dist_matrix)
                        #normalized_distance = find_nearest_distance_between_clusters(cluster1, cluster2,dist_matrix)
                        
                        if (normalized_distance >= norm_dist_threshold):
                            continue
                        rank_order_distance = find_rank_order(cluster1, cluster2)
                        if (rank_order_distance >= rank_threshold):
                            continue
                        G.add_edge(cluster1, cluster2)
            #print()     
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
        #distanceThreshold=(norm_dist_threshold,rank_threshold)
        matched_clusters, unmatched_clusters = find_clusters(faces,dist_matrix,distanceThreshold[0],distanceThreshold[1])
        #print('matched_len:',len(matched_clusters),'unmatched_len:',len(unmatched_clusters))
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

class FeatureExtractor:
    def __init__(self,vggmodel=None):
        if vggmodel is None:
            self.tfInference=TensorFlowInference('age_gender_tf2_new-01-0.14-0.92.pb',input_tensor='input_1:0',output_tensor='global_pooling/Mean:0')
        else:
            self.tfInference=None
            
            from keras_vggface.vggface import VGGFace
            from keras.engine import  Model
            layers={'vgg16':'fc7/relu','resnet50':'avg_pool'}
            model = VGGFace(model=vggmodel)
            out = model.get_layer(layers[vggmodel]).output
            self.cnn_model = Model(model.input, out)
            _,w,h,_=model.input.shape
            self.size=(int(w),int(h))
            
    def extract_features(self,image_path):
        if self.tfInference is not None:
            return self.tfInference.extract_features(image_path)
        else:
            from keras_vggface.utils import preprocess_input
            from keras.preprocessing import image
            img = image.load_img(image_path, target_size=self.size)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            preds = self.cnn_model.predict(x).reshape(-1)
            return preds
    
    def close(self):
        if self.tfInference is not None:
            self.tfInference.close_session()


#B-cubed
def fscore(p_val, r_val, beta=1.0):
    """Computes the F_{beta}-score of given precision and recall values."""
    return (1.0 + beta**2) * (p_val * r_val / (beta**2 * p_val + r_val))
def mult_precision(el1, el2, cdict, ldict):
    """Computes the multiplicity precision for two elements."""
    return min(len(cdict[el1] & cdict[el2]), len(ldict[el1] & ldict[el2])) \
        / float(len(cdict[el1] & cdict[el2]))

def mult_recall(el1, el2, cdict, ldict):
    """Computes the multiplicity recall for two elements."""
    return min(len(cdict[el1] & cdict[el2]), len(ldict[el1] & ldict[el2])) \
        / float(len(ldict[el1] & ldict[el2]))

def precision(cdict, ldict):
    """Computes overall extended BCubed precision for the C and L dicts."""
    return np.mean([np.mean([mult_precision(el1, el2, cdict, ldict) \
        for el2 in cdict if cdict[el1] & cdict[el2]]) for el1 in cdict])

def recall(cdict, ldict):
    """Computes overall extended BCubed recall for the C and L dicts."""
    return np.mean([np.mean([mult_recall(el1, el2, cdict, ldict) \
        for el2 in cdict if ldict[el1] & ldict[el2]]) for el1 in cdict])
        

def get_BCubed_set(y_vals):
    dic={}
    for i,y in enumerate (y_vals):
        dic[i]=set([y])
    return dic

def BCubed_stat(y_true, y_pred, beta=1.0):
    cdict=get_BCubed_set(y_true)
    ldict=get_BCubed_set(y_pred)
    p=precision(cdict, ldict)
    r=recall(cdict, ldict)
    f=fscore(p, r, beta)
    return (p,r,f)
    
featureExtractor=None
def get_clustering_results(db_dir,method,distanceThreshold):
    features_file=os.path.join(db_dir,'features%s.npz'%(model_desc[model_ind][1]))
    #features_file='D:/src_code/HSE_FaceRec_tf/lfw_ytf_subset_resnet_feats_vgg2.npz'
    if not os.path.exists(features_file):
        print(db_dir)
        global featureExtractor
        if featureExtractor is None:
            featureExtractor=FeatureExtractor(model_desc[model_ind][0])
        
        dirs_and_files=np.array([[d,os.path.join(d,f)] for d in next(os.walk(db_dir))[1] for f in next(os.walk(os.path.join(db_dir,d)))[2] if is_image(f)])
        #dirs_and_files=np.array([[d,os.path.join(d,f)] for d in next(os.walk(db_dir))[1] if d!='1' and d!='2' for f in next(os.walk(os.path.join(db_dir,d)))[2] if is_image(f)])
        dirs=dirs_and_files[:,0]
        files=dirs_and_files[:,1]

        label_enc=preprocessing.LabelEncoder()
        label_enc.fit(dirs)
        y_true=label_enc.transform(dirs)
        #print ('y=',y)
        start_time = time.time()
        X=np.array([featureExtractor.extract_features(os.path.join(db_dir,filepath)) for filepath in files])
        np.savez(features_file,x=X,y_true=y_true)
    
    data = np.load(features_file)
    X=data['x']
    X_norm=preprocessing.normalize(X,norm='l2')
    y_true=data['y_true']
    #y_true=data['y']
    
    label_enc=preprocessing.LabelEncoder()
    label_enc.fit(y_true)
    y_true=label_enc.transform(y_true)
    
    num_features=X_norm.shape[1]
    #print('num_samples=',X_norm.shape[0],'num_features=',num_features)
    pair_dist=pairwise_distances(X_norm)#/num_features
    
    global clusteringMethod
    clusteringMethod=method
    clusters=get_facial_clusters(pair_dist,distanceThreshold)
    
    y_pred=-np.ones(len(y_true))
    for ind,cluster in enumerate(clusters):
        y_pred[cluster]=ind
    ind=len(clusters)
    for i in range(len(y_pred)):
        if y_pred[i]==-1:
            ind+=1
            y_pred[i]=ind

    num_of_classes=len(np.unique(y_true))
    num_of_clusters=len(clusters)
    #print('X.shape:',X_norm.shape,'num of classes:',num_of_classes,'num of clusters:',num_of_clusters)
    return num_of_classes,num_of_clusters,y_true, y_pred    

def get_clustering_statistics(db_dir,method,distanceThreshold):
    num_of_classes,num_of_clusters,y_true, y_pred=get_clustering_results(db_dir,method,distanceThreshold)
    ari=metrics.adjusted_rand_score(y_true, y_pred)
    ami=metrics.adjusted_mutual_info_score(y_true, y_pred,average_method ='arithmetic')
    homogeneity,completeness,v_measure=metrics.homogeneity_completeness_v_measure(y_true, y_pred)
    #fm=metrics.fowlkes_mallows_score(y_true, y_pred)
    bcubed_precision,bcubed_recall,bcubed_fmeasure=BCubed_stat(y_true, y_pred)
    return num_of_classes,num_of_clusters,ari,ami,homogeneity,completeness,v_measure,bcubed_precision,bcubed_recall,bcubed_fmeasure

def test_clustering(db_dir,method,distanceThreshold):
    num_of_classes,num_of_clusters,ari,ami,homogeneity,completeness,v_measure,bcubed_precision,bcubed_recall,bcubed_fmeasure=get_clustering_statistics(db_dir,method,distanceThreshold)
    print('adjusted_rand_score:',ari)
    print('Adjusted Mutual Information:',ami)
    print('homogeneity/completeness/v-measure:',homogeneity,completeness,v_measure)
    print('BCubed precision/recall/FMeasure:',bcubed_precision,bcubed_recall,bcubed_fmeasure)
    #print('Fowlkes-Mallows index:',fm)

def test_avg_clustering(db_dirs,method,distanceThreshold):
    num_of_dirs=len(db_dirs)
    stats_names=['classes','clusters','ARI','AMI','homogeneity','completeness','v-measure','BCubed_precision','BCubed_recall','BCubed_FMeasure']
    stats=np.zeros((num_of_dirs,len(stats_names)))
    for i,db_dir in enumerate(db_dirs):
        stats[i]=get_clustering_statistics(db_dir,method,distanceThreshold)
        
        
    mean_stats=np.mean(stats,axis=0)
    std_stats=np.std(stats,axis=0)
    for i,stat in enumerate(stats_names):
        print('%s:%.3f(%.3f) '%(stat,mean_stats[i],std_stats[i]), end='')
    print('\n')

def test_avg_clustering_with_model_selection(db_dirs,method,val_dirs_count=2):
    bestStatistic,prevStatistic=0,0
    val_dirs_count=len(db_dirs) #hack!!!
    if use_clustering==rankorder_clustering:
        bestThreshold=(0,0)
        for distanceThreshold in np.linspace(1.02,1.1,9):
            prevStatistic=0
            bestChanged=False
            for rankThreshold in range(12,22,2):
                currentStatistic=0
                for i,db_dir in enumerate(db_dirs[:val_dirs_count]):
                    num_of_classes,num_of_clusters,y_true, y_pred=get_clustering_results(db_dir,method,(distanceThreshold,rankThreshold))
                    #bcubed_precision,bcubed_recall,bcubed_fmeasure=BCubed_stat(y_true, y_pred)
                    #currentStatistic+=bcubed_fmeasure
                    homogeneity,completeness,v_measure=metrics.homogeneity_completeness_v_measure(y_true, y_pred)
                    currentStatistic+=v_measure
                    #print(num_of_classes)
                currentStatistic/=val_dirs_count
                print(distanceThreshold,rankThreshold,currentStatistic)
                if currentStatistic>bestStatistic:
                    bestStatistic=currentStatistic
                    bestThreshold=(distanceThreshold,rankThreshold)
                    bestChanged=True
                if currentStatistic<=prevStatistic: #-0.01
                    break
                prevStatistic=currentStatistic
            if not bestChanged:
                break
    else:
        bestThreshold=0
        for distanceThreshold in np.linspace(0.6,1.3,71):
            currentStatistic=0
            for i,db_dir in enumerate(db_dirs[:val_dirs_count]):
                num_of_classes,num_of_clusters,y_true, y_pred=get_clustering_results(db_dir,method,distanceThreshold)
                bcubed_precision,bcubed_recall,bcubed_fmeasure=BCubed_stat(y_true, y_pred)
                currentStatistic+=bcubed_precision
                #homogeneity,completeness,v_measure=metrics.homogeneity_completeness_v_measure(y_true, y_pred)
                #currentStatistic+=v_measure
                #print(num_of_classes)
            currentStatistic/=val_dirs_count
            #print(distanceThreshold,currentStatistic)
            if currentStatistic>bestStatistic:
                bestStatistic=currentStatistic
                bestThreshold=distanceThreshold
            if currentStatistic<prevStatistic-0.01:
                break
            if currentStatistic>0.85:
                break
            prevStatistic=currentStatistic
        
    print('method:',method,'bestParams:',bestThreshold,'bestStatistic:',bestStatistic)
    #test_avg_clustering(db_dirs[val_dirs_count:],method,bestThreshold)
    test_avg_clustering(db_dirs,method,bestThreshold)  #hack!!!

model_desc=[[None,''],['vgg16','_vgg16'],['resnet50','_resnet50']]
model_ind=0
if __name__ == '__main__':
    db_dirs=[]
    if True:
        db_dirs.append('D:/datasets/my_photos/GallagherDataset/faces')
    else:
        for i in range(0,58):
            db_dirs.append('D:/datasets/my_photos/GFW_release/%d'%(i))
    if use_clustering==rankorder_clustering:
        method_threshold_list=[['single',(0.9,14)]]
    else:
        method_threshold_list=[['single',0.78],['average',0.96]]
        #method_threshold_list=[['single',0.78],['average',0.96],['complete',1.1],['weighted',1],['centroid',1],['median',1],['ward',1]]
        #method_threshold_list=[['single',0.00076],['average',0.00094],['complete',0.00107]]
    if False:
        #ind=0
        #method=method_threshold_list[ind][0]
        #distanceThreshold=method_threshold_list[ind][1]

        for method,distanceThreshold in method_threshold_list:
            print('method:',method)
            test_avg_clustering(db_dirs,method,distanceThreshold)
    else:
        for method,_ in method_threshold_list:
            test_avg_clustering_with_model_selection(db_dirs,method)
    
    if featureExtractor is not None:
        featureExtractor.close()