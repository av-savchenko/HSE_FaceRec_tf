from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time,datetime
import cv2
import shutil
import pickle
import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from configparser import ConfigParser

from facial_analysis import FacialImageProcessing, get_video_file_orientation,is_image,is_video
from facial_clustering import get_facial_clusters

    
#config values
minDaysDifferenceBetweenPhotoMDates=2
minNoPhotos=2
minNoFrames=10
distanceThreshold=0.82
minFaceWidthPercent=0.05


img_size=224
def process_image(imgProcessing,img):
    height, width, channels = img.shape
    bounding_boxes, _,ages,genders,facial_features=imgProcessing.process_image(img)
    facial_images=[]
    has_center_face=False
    for bb in bounding_boxes:
        x1,y1,x2,y2=bb[0:4]
        face_img=cv2.resize(img[y1:y2,x1:x2,:],(img_size,img_size))
        facial_images.append(face_img)
        #dx=1.5*(x2-x1)
        if (x2-x1)/width>=minFaceWidthPercent: #x1-dx<=width/2<=x2+dx:
            has_center_face=True
    return facial_images,ages,genders,facial_features, has_center_face

def perform_clustering(mdates,all_indices,all_features,all_born_years,no_images_in_cluster, checkDates=True):
    def feature_distance(i,j):
        dist=np.sqrt(np.sum((all_features[i]-all_features[j])**2))
        max_year=max(mdates[all_indices[i]].tm_year,mdates[all_indices[j]].tm_year)
        cur_age_i,cur_age_j=max_year-all_born_years[i],max_year-all_born_years[j]
        age_dist=(cur_age_i-cur_age_j)**2/(cur_age_i+cur_age_j)
        return [dist,age_dist*0.1]
        
    num_faces=len(all_indices)
    if num_faces<no_images_in_cluster:
        return []

    t=time.time()
    pair_dist=np.array([[feature_distance(i,j) for j in range(num_faces)] for i in range(num_faces)])
    dist_matrix=np.clip(np.sum(pair_dist,axis=2),a_min=0,a_max=None)
    clusters=get_facial_clusters(dist_matrix,distanceThreshold,all_indices,no_images_in_cluster)
    elapsed = time.time() - t
    #print('clustering elapsed=%f'%(elapsed)) 
    
    print('clusters',clusters)
    
    def is_good_cluster(cluster):
        res=len(cluster)>=no_images_in_cluster
        if res and checkDates:
            cluster_mdates=[mdates[all_indices[i]] for i in cluster]
            max_date,min_date=max(cluster_mdates),min(cluster_mdates)
            diff_in_days=(datetime.datetime.fromtimestamp(time.mktime(max_date))-datetime.datetime.fromtimestamp(time.mktime(min_date))).days
            res=diff_in_days>=minDaysDifferenceBetweenPhotoMDates
        return res
        
    filtered_clusters=[cluster for cluster in clusters if is_good_cluster(cluster)]

    return filtered_clusters

    
def process_video(imgProcessing,video_filepath,mdate):
    video_year=mdate.tm_year+(mdate.tm_mon-1)/12
    mdates=[]

    rotation=get_video_file_orientation(video_filepath)
    counter=0
    delta_counter=5
    video = cv2.VideoCapture(video_filepath)
    t=time.time()
    all_facial_images,all_born_years, all_genders,all_features,all_normed_features,all_indices=[],[],[],[],[],[]
    
    frame_count=0
    while video.isOpened():
        if video.grab()==0:
            break
        counter+=1
        if counter%delta_counter!=0:
            continue
        _,draw=video.retrieve()
        height, width, channels = draw.shape
        #if width>640 or height>480:
        #    draw=cv2.resize(draw, (min(width,640),min(height,480)))
        if rotation==90:
            draw=cv2.transpose(draw)
            draw=cv2.flip(draw,1)
        elif rotation==270:
            draw=cv2.transpose(draw)
            draw=cv2.flip(draw,0)
        facial_images,ages,genders,facial_features,has_center_face=process_image(imgProcessing,draw)
        all_facial_images.extend(facial_images)
        all_genders.extend(genders)
        all_features.extend(facial_features)
        for features in facial_features:
            all_normed_features.append(features/np.sqrt(np.sum(features**2)))
        all_indices.extend([frame_count]*len(ages))
        mdates.append(mdate)
        all_born_years.extend([(video_year-(age-0.5)) for age in ages])
        frame_count+=1
        delta_counter=5 if len(ages)==0 else 3
        
    elapsed = time.time() - t

    all_born_years=np.array(all_born_years)
    all_genders=np.array(all_genders)
    all_features=np.array(all_features)
    all_normed_features=np.array(all_normed_features)
    
    print('\nvideo %s processing elapsed=%f'%(os.path.basename(video_filepath),elapsed))            
    
    filtered_clusters=perform_clustering(mdates,all_indices,all_normed_features,all_born_years,minNoFrames, checkDates=False)
    
    if False:
        no_clusters=min(10,len(filtered_clusters))
        plt_ind=1
        for i in range(no_clusters):
            l=len(filtered_clusters[i])
            step=l//minNoFrames
            for j in range(0,step*minNoFrames,step):
                plt.subplot(no_clusters,minNoFrames,plt_ind)
                plt.imshow(cv2.cvtColor(all_facial_images[filtered_clusters[i][j]],cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt_ind+=1

        plt.show()
    
    cluster_facial_images,cluster_ages,cluster_genders,cluster_facial_features=[],[],[],[]
    for i,cluster in enumerate(filtered_clusters):
        avg_gender_preds=np.median(all_genders[cluster])
        avg_year=np.median(all_born_years[cluster])
        print('cluster ',i,avg_gender_preds,avg_year)
        cluster_facial_images.append(all_facial_images[cluster[0]])
        cluster_genders.append(avg_gender_preds)
        cluster_ages.append(int(video_year-(avg_year-0.5)))
        cluster_facial_features.append(np.mean(all_features[cluster],axis=0))
    
    video_has_faces=len(filtered_clusters)>0
    return cluster_facial_images,cluster_ages,cluster_genders,cluster_facial_features,video_has_faces

    
#Dempster-SHafer implementation
def calculate_proximity(dt, predictions, num_of_classes=100):
    prox_classes = []
    for i in range(0,num_of_classes):
        class_preds = predictions
        class_dt = dt[i]
        norm_vect = np.power((1+LA.norm(np.subtract(class_dt, class_preds))), -1)
        #norm_vect = np.power((1 + np.sum(abs((np.subtract(class_dt, class_preds))), axis=0)), -1)
        prox_classes.append(norm_vect)
    norm_prox_classes = prox_classes/sum(prox_classes)
    return norm_prox_classes


def compute_belief_degrees(proximities, num_of_classes):
    belief_degrees = []
    current_classifier_prox = proximities
    for j in range(0, num_of_classes):
        class_mult = [(1-current_classifier_prox[k]) for k in range(0, num_of_classes) if k != j]
        num = (current_classifier_prox[j] * np.prod(class_mult))
        denom = (1 - current_classifier_prox[j])*(1-np.prod(class_mult))
        cl_ev = num / denom
        belief_degrees.append(cl_ev)
        print(np.sum(belief_degrees))
    return belief_degrees

def compute_b(proximities, num_of_classes):
    belief_degrees = []
    for j in range(0, num_of_classes):
        class_mult = [(1-proximities[k]) for k in range(0, num_of_classes) if k != j]
        #num = (proximities[j] * np.prod(class_mult))
        #denom = 1 - proximities[j]*(1-np.prod(class_mult))
        #cl_ev = (num / denom)
        num = np.log(proximities[j]) + np.sum(np.log(class_mult))
        denom = np.log(1-proximities[j]*(1-np.prod(class_mult)))
        cl_ev = num-denom
        belief_degrees.append(cl_ev)
    return belief_degrees

def final_decision(log_belief_degrees):
    #belief_degrees = np.log(np.asarray(belief_degrees))
   # belief_degrees = np.exp(np.log(np.asarray(belief_degrees)))
    #m = np.prod(belief_degrees, axis=0, dtype=np.float32)
    log_m = np.sum(log_belief_degrees, axis=0)
    #m = np.exp(log_m)
    m=log_m
    #print(m)
    index = m.argsort()[::-1][:1]
    return index[0]

def dempster_shafer_gender(male_probabs):
    dt=[[0.875,0.125],[0.353,0.647]]
    beliefs=[]
    for male_probab in male_probabs:
        gender_preds = [[male_probab[0],1-male_probab[0]]]
        gender_proximities = calculate_proximity(dt, gender_preds, 2)
        b = compute_b(gender_proximities,2)
        beliefs.append(b)
    ds_gender = final_decision(beliefs)
    return ds_gender

def process_album(imgProcessing,album_dir):
    features_file=os.path.join(album_dir,'features.dump')
    t=time.time()
    if os.path.exists(features_file):
        with open(features_file, "rb") as f:
            files=pickle.load(f)
            mdates=pickle.load(f)
            all_facial_images=pickle.load(f)
            all_born_years=pickle.load(f)
            all_genders=pickle.load(f)
            all_features=pickle.load(f)
            all_indices=pickle.load(f)
            private_photo_indices=pickle.load(f)
    else:
        #process static images
        files=[f for f in next(os.walk(album_dir))[2] if is_image(f)]
        #files=files[:20]
        mdates=[time.gmtime(os.path.getmtime(os.path.join(album_dir,f))) for f in files]
        all_facial_images,all_born_years, all_genders,all_features,all_indices,private_photo_indices=[],[],[],[],[],[]
        for i,fpath in enumerate(files):
            full_photo = cv2.imread(os.path.join(album_dir,fpath))
            facial_images,ages,genders,facial_features,has_center_face=process_image(imgProcessing,full_photo)
            if len(facial_images)==0:
                full_photo_t=cv2.transpose(full_photo)
                rotate90=cv2.flip(full_photo_t,1)
                facial_images,ages,genders,facial_features,has_center_face=process_image(imgProcessing,rotate90)
                if len(facial_images)==0:
                    rotate270=cv2.flip(full_photo_t,0)
                    facial_images,ages,genders,facial_features,has_center_face=process_image(imgProcessing,rotate270)
            if has_center_face:
                private_photo_indices.append(i)
            all_facial_images.extend(facial_images)
            all_genders.extend(genders)
            for features in facial_features:
                features=features/np.sqrt(np.sum(features**2))
                all_features.append(features)
            all_indices.extend([i]*len(ages))

            photo_year=mdates[i].tm_year+(mdates[i].tm_mon-1)/12
            all_born_years.extend([(photo_year-(age-0.5)) for age in ages])
            
            print ('Processed photos: %d/%d\r'%(i+1,len(files)),end='')
            sys.stdout.flush()
            

        with open(features_file, "wb") as f:
            pickle.dump(files,f)
            pickle.dump(mdates, f)
            pickle.dump(all_facial_images,f)
            pickle.dump(all_born_years, f)
            pickle.dump(all_genders,f)
            pickle.dump(all_features,f)
            pickle.dump(all_indices,f)
            pickle.dump(private_photo_indices, f)
        print('features dumped into',features_file)

    elapsed = time.time() - t
    no_image_files=len(files)
    print('\nelapsed=%f for processing of %d files'%(elapsed,no_image_files))            

    #process video files
    video_files=[f for f in next(os.walk(album_dir))[2] if is_video(f)]
    #video_files=[]
    video_mdates=[time.gmtime(os.path.getmtime(os.path.join(album_dir,f))) for f in video_files]
    t=time.time()
    for i,fpath in enumerate(video_files):
        video_filepath=os.path.join(album_dir,fpath)
        facial_images,ages,genders,facial_features,has_center_face=process_video(imgProcessing,video_filepath,video_mdates[i])
        if has_center_face:
            private_photo_indices.append(i+no_image_files)
        all_facial_images.extend(facial_images)
        all_genders.extend(genders)
        for features in facial_features:
            features=features/np.sqrt(np.sum(features**2))
            all_features.append(features)
        all_indices.extend([i+no_image_files]*len(ages))

        photo_year=video_mdates[i].tm_year+(video_mdates[i].tm_mon-1)/12
        all_born_years.extend([(photo_year-(age-0.5)) for age in ages])
            
    elapsed = time.time() - t

    files.extend(video_files)
    mdates.extend(video_mdates)
    all_born_years=np.array(all_born_years)
    all_genders=np.array(all_genders)
    all_features=np.array(all_features)

    print('\nelapsed=%f for processing of %d videos'%(elapsed,len(video_files)))         
    print(all_born_years.shape,all_genders.shape,all_features.shape,len(all_indices))
    
    filtered_clusters=perform_clustering(mdates,all_indices, all_features, all_born_years,minNoPhotos)
    
    for video_id in range(no_image_files,len(files)):
        for face_ind in (ind for ind,file_id in enumerate(all_indices) if file_id==video_id):
            found=False
            for i,cluster in enumerate(filtered_clusters):
                if face_ind in cluster:
                    print(face_ind, ' for video ',video_id,' in cluster ', i)
                    found=True
                    break
            if not found:
                print('cluster for ',face_ind,' in video ',video_id,' not found')
    
    cluster_genders,cluster_ages=[],[]
    for i,cluster in enumerate(filtered_clusters):
        avg_gender_preds=np.median(all_genders[cluster])
        avg_year=np.median(all_born_years[cluster])
        ds_gender=dempster_shafer_gender(all_genders[cluster])
        print('cluster ',i,avg_gender_preds,ds_gender,avg_year)
        #cluster_genders.append('male' if avg_gender_preds>=0.6 else 'female')
        cluster_genders.append('male' if ds_gender==0 else 'female')
        cluster_ages.append(int(avg_year))

    if True:
        res_dir=os.path.join(album_dir,'clusters')
        if os.path.exists(res_dir):
            shutil.rmtree(res_dir,ignore_errors=True)
            time.sleep(2)
        for i,cluster in enumerate(filtered_clusters):
            clust_dir=os.path.join(res_dir,'%d %s %d'%(i,cluster_genders[i],cluster_ages[i]))
            os.makedirs(clust_dir)
            for ind in cluster:
                cv2.imwrite(os.path.join(clust_dir,'%d.jpg'%(ind)),all_facial_images[ind])

        if True:
            private_photos=set([all_indices[elem] for cluster in filtered_clusters for elem in cluster]) | set(private_photo_indices)
            public_photo_path=((i,fpath) for i,fpath in enumerate(files) if i not in private_photos)
            dst_dir=os.path.join(res_dir,'public')
            os.makedirs(dst_dir)
            for i,fpath in public_photo_path:
                
                if i<no_image_files:
                    full_photo = cv2.imread(os.path.join(album_dir,fpath))
                    r = 200.0 / full_photo.shape[1]
                    dim = (200, int(full_photo.shape[0] * r))
                    full_photo=cv2.resize(full_photo, dim)
                    cv2.imwrite(os.path.join(dst_dir,fpath),full_photo)
                else:
                    shutil.copy(os.path.join(album_dir,fpath),dst_dir)
        
    if True:
        no_clusters=min(10,len(filtered_clusters))
        plt_ind=1
        for i in range(no_clusters):
            for j in range(minNoPhotos):
                plt.subplot(no_clusters,minNoPhotos,plt_ind)
                plt.imshow(cv2.cvtColor(all_facial_images[filtered_clusters[i][j]],cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt_ind+=1

        plt.show()
    

if __name__ == '__main__':
    config = ConfigParser()
    config.read('config.txt')
    default_config=config['DEFAULT']
    minDaysDifferenceBetweenPhotoMDates=int(default_config['MinDaysDifferenceBetweenPhotoMDates'])
    minNoPhotos=int(default_config['MinNoPhotos'])
    minNoFrames=int(default_config['MinNoFrames'])
    distanceThreshold=float(default_config['DistanceThreshold'])
    minFaceWidthPercent=float(default_config['MinFaceWidthPercent'])/100
    
    print('minDaysDifferenceBetweenPhotoMDates:',minDaysDifferenceBetweenPhotoMDates,' minNoPhotos:',minNoPhotos,'minNoFrames:',minNoFrames,' distanceThreshold:',distanceThreshold,' minFaceWidthPercent:',minFaceWidthPercent)
    
    imgProcessing=FacialImageProcessing(print_stat=False,minsize = 112)
    process_album(imgProcessing,default_config['InputDirectory'])
    #video_filepath='D:/datasets/my_photos/iphone/IMG_2220.MOV' #'D:/datasets/my_photos/iphone/video.AVI'
    #process_video(imgProcessing,video_filepath,time.gmtime(os.path.getmtime(video_filepath))) 
    imgProcessing.close()
