import argparse
import sys
import os.path
import os
import datetime, time
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from PIL import Image


import numpy as np
np.random.seed(123)  # for reproducibility


use_my_cnn=True
use_keras=False
use_lfw=True

img_extensions=['.jpg','.jpeg','.png']
def is_image(path):
    _, file_extension = os.path.splitext(path)
    return file_extension.lower() in img_extensions

def get_files(db_dir):
    return [[d,os.path.join(d,f)] for d in next(os.walk(db_dir))[1] for f in next(os.walk(os.path.join(db_dir,d)))[2] if is_image(f)]

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ''
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

def load_graph(frozen_graph_filename, prefix=''):
    with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name=prefix)
    return graph
    
class TensorFlowInference:
    def __init__(self,frozen_graph_filename,input_tensor,output_tensor,learning_phase_tensor=None):
        graph = load_graph(frozen_graph_filename,'')
        print([n.name for n in graph.as_graph_def().node if 'input' in n.name])
        
        graph_op_list=list(graph.get_operations())
        print([n.name for n in graph_op_list if 'keras_learning' in n.name])
        
        self.tf_sess=tf.Session(graph=graph)
        
        self.tf_input_image = graph.get_tensor_by_name(input_tensor)
        print('tf_input_image=',self.tf_input_image)
        self.tf_output_features = graph.get_tensor_by_name(output_tensor)
        print('tf_output_features=',self.tf_output_features)
        self.tf_learning_phase = graph.get_tensor_by_name(learning_phase_tensor) if learning_phase_tensor else None;
        print('tf_learning_phase=',self.tf_learning_phase)
        _,w,h,_=self.tf_input_image.shape
        self.w,self.h=int(w),int(h)
        print ('input w,h',self.w,self.h,' output shape:',self.tf_output_features.shape)
        
    def extract_features(self,img_filepath):
        if False:# or not use_lfw:
            img = image.load_img(img_filepath, target_size=(self.w,self.h))#(224, 224))
        else:
            orig_w,orig_h=250,250
            img = image.load_img(img_filepath,target_size=(orig_w,orig_h))
            w1,h1=128,128
            dw=(orig_w-w1)/2
            dh=(orig_h-h1)/2
            box = (dw, dh, orig_w-dw, orig_h-dh)
            img = img.crop(box)
            img = img.resize((self.w,self.h))
        
        x = image.img_to_array(img)
        # 'RGB'->'BGR'
        x = x[..., ::-1]
        # Zero-center by mean pixel
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68
        x = np.expand_dims(x, axis=0)
        feed_dict={self.tf_input_image: x}
        if self.tf_learning_phase is not None:
            feed_dict[self.tf_learning_phase]=0
        preds = self.tf_sess.run(self.tf_output_features, feed_dict=feed_dict).reshape(-1)
        return preds
    
    def extract_features_from_images(self,image_dir, filepath_list):      
        res=None
        feed_dict={}
        if self.tf_learning_phase is not None:
            feed_dict[self.tf_learning_phase]=0
        images_no=len(filepath_list)
        batch_size=32
        for ndx in range(0, images_no, batch_size):
            images=[]
            file_list= filepath_list[ndx:min(ndx + batch_size, images_no)]
            for filepath in file_list:
                img_filepath=os.path.join(db_dir,filepath)
                if False:# or not use_lfw:
                    img = image.load_img(img_filepath, target_size=(self.w,self.h))#(224, 224))
                else:
                    orig_w,orig_h=250,250
                    img = image.load_img(img_filepath,target_size=(orig_w,orig_h))
                    w1,h1=128,128
                    dw=(orig_w-w1)/2
                    dh=(orig_h-h1)/2
                    box = (dw, dh, orig_w-dw, orig_h-dh)
                    img = img.crop(box)
                    img = img.resize((self.w,self.h))
                
                x = image.img_to_array(img)
                # 'RGB'->'BGR'
                x = x[..., ::-1]
                # Zero-center by mean pixel
                #x = np.expand_dims(x, axis=0)
                images.append(x)
            images=np.array(images)
            images[..., 0] -= 103.939
            images[..., 1] -= 116.779
            images[..., 2] -= 123.68
            #print ('images:',images.shape)
            feed_dict[self.tf_input_image]=images
            preds = self.tf_sess.run(self.tf_output_features, feed_dict=feed_dict).reshape(images.shape[0],-1)
            #print(x.shape,preds.shape)
            if res is None:
                res=preds
            else:
                res=np.vstack((res,preds))
        
        res=np.array(res)
        #print ('res:',res.shape)
        return res
        
    def close_session(self):
        self.tf_sess.close()


def extract_keras_features(model,img_filepath):
    _,w,h,_=model.input.shape
    w,h=int(w),int(h)
    if True:# or not use_lfw:
        img = image.load_img(img_filepath, target_size=(w,h))#(224, 224))
    else:
        orig_w,orig_h=250,250
        img = image.load_img(img_filepath,target_size=(orig_w,orig_h))
        w1,h1=128,128
        dw=(orig_w-w1)/2
        dh=(orig_h-h1)/2
        box = (dw, dh, orig_w-dw, orig_h-dh)
        img = img.crop(box)
        img = img.resize((w,h))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x).reshape(-1)
    return preds

def convert_features_to_text(X,y,filepath):
    with open(filepath,'w') as outfile:
        for i in range(len(y)):
            outfile.write('{0}.jpg\n{0}\n'.format(y[i]))
            for feat in X[i,:]:
                outfile.write('{0:g} '.format(feat))
            outfile.write('\n')

def save_video_features_to_text(db_dir,filepath, nn_model):
    top_dirs=[d for d in next(os.walk(db_dir))[1]]
    
    start_time = time.time()
    with open(filepath,'w') as outfile:
        for i,top_dir in enumerate(top_dirs):
            print(top_dir)
            #if i>3:
            #    break
            top_dir_path=os.path.join(db_dir,top_dir)
            dirs=[os.path.join(top_dir_path,d) for d in next(os.walk(top_dir_path))[1]]
            outfile.write('{0}\n{1}\n'.format(top_dir,len(dirs)))
            for video in dirs:
                files=[os.path.join(video,f) for f in next(os.walk(video))[2] if is_image(f)]
                files=files[0:len(files):5]
                outfile.write('{0}\n'.format(len(files)))
                for f in files:
                    outfile.write(f+'\n')
                    if use_keras:
                        X=extract_keras_features(nn_model,f)
                    else:
                        X=nn_model.extract_features(f)
                    for feat in X:
                        outfile.write('{0:g} '.format(feat))
                    outfile.write('\n')
    if not use_keras:
        nn_model.close_session()
    print('--- %s seconds ---' % (time.time() - start_time))
    sys.exit()

def chi2dist(x, y):
    sum=x+y
    chi2array=np.where(sum>0, (x-y)**2/sum, 0)
    return np.sum(chi2array)

def KL_dist(x, y):
    KL_array=(x+0.001)*np.log((x+0.001)/(y+0.001))
    return np.sum(KL_array)

def classifier_tester(classifier,x,y):
    sss=model_selection.StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
    scores=model_selection.cross_validate(classifier,x, y, scoring='accuracy',cv=sss)
    acc=scores['test_score']
    print('accuracies=',acc*100)
    print('total acc=',round(acc.mean()*100,2),round(acc.std()*100,2))
    print('test time=',scores['score_time'])

if __name__ == '__main__':
    if use_lfw:
        #features_file='lfw_mobilenet_feats_224-09-new.npz'
        #features_file='lfw_mobilenet_feats_224-03.npz'
        #features_file='lfw_mobilenet2_feats_224-05.npz'
        #features_file='lfw_nasnet_feats_224-08.npz'
        features_file='lfw_mobilenet_vgg2_feats_192.npz'
        #features_file='lfw_mobilenet_feats_vgg2_resnet.npz'
        #features_file='lfw_nasnet_features.npz'
        db_dir='D:/datasets/lfw_ytf/lfw'#-deepfunneled' lfw_cropped
        test_size=0.5
    elif False:
        features_file='ijba_mobilenet_feats_192.npz'
        db_dir='../images/IJBA/1N_images/still'
        test_size=0.5
    else:
        features_file='pf83_mymobilenet_feats_192.npz'
        db_dir='../images/pubfig83/saved_color'
        test_size=0.05

    save_video_features=False
    if not os.path.exists(features_file) or save_video_features:
        os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

        import tensorflow as tf
        if use_keras:
            from keras.engine import  Model
            from keras.preprocessing import image
            from keras import backend as K

            if use_my_cnn:
                from keras.layers import Flatten, Dense, Dropout,GlobalAveragePooling2D, Reshape, Conv2D, Activation
                from keras.models import Model
                from keras.applications.imagenet_utils import preprocess_input
                from keras.applications import VGG16, VGG19, ResNet50, Xception, MobileNet

                classes_num=9131 #10575
                sz=192
                net_model = MobileNet(weights=None, include_top=False,
                          input_shape=(sz, sz, 3))
                last_model_layer = net_model.output
                x = GlobalAveragePooling2D()(last_model_layer)
                x = Reshape((1,1,1024), name='reshape_1')(x)
                if False:
                    x = Conv2D(classes_num, (1, 1),
                               padding='same', name='conv_preds')(x)
                    x = Activation('softmax', name='act_softmax')(x)
                    preds = Reshape((classes_num,), name='reshape_2')(x)
                    model = Model(net_model.input, preds)
                    model.load_weights('models/vgg2_mobilenet192_weights-improvement-0.95.hdf5')
                else:
                    model = Model(net_model.input, x)
                    model.load_weights('models/vgg2_mobilenet.h5')
                out = model.get_layer('reshape_1').output
                print('out=',out)
            else:
                from keras_vggface.vggface import VGGFace
                from keras_vggface.utils import preprocess_input
                model = VGGFace() # pooling: None, avg or max
                out = model.get_layer('fc7/relu').output
            
            cnn_model = Model(model.input, out)
            #cnn_model.save('models/vgg2_mobilenet.h5')
            #frozen_graph = freeze_session(K.get_session(), output_names=[cnn_model.output.op.name])
            #tf.train.write_graph(frozen_graph, 'models', 'vgg2_mobilenet.pb', as_text=False)
        else:
            import tensorflow.contrib.keras as keras
            from keras.preprocessing import image
            #tfInference=TensorFlowInference('../DNN_models/my_tf/tf_vgg2_nasnet-09.pb',input_tensor='images:0',output_tensor='final_layer/Mean:0')
            tfInference=TensorFlowInference('../DNN_models/my_tf/vgg2_mobilenet.pb',input_tensor='input_1:0',output_tensor='reshape_1/Reshape:0',learning_phase_tensor='conv1_bn/keras_learning_phase:0')
            #tfInference=TensorFlowInference('../DNN_models/my_tf/vgg2_mobilenet224-08-0.94.pb',input_tensor='input_1:0',output_tensor='reshape_1/Reshape:0',learning_phase_tensor='conv1_bn/keras_learning_phase:0')
            #tfInference=TensorFlowInference('../DNN_models/my_tf/vgg2_mobilenet224-09-0.94.pb',input_tensor='input_1:0',output_tensor='reshape_1/Reshape:0')
            
            #tfInference=TensorFlowInference('../DNN_models/my_tf/age_gender_tf2_new-01-0.14-0.92.pb',input_tensor='input_1:0',output_tensor='global_pooling/Mean:0')
            #tfInference=TensorFlowInference('../DNN_models/my_tf/tf_vgg2_mobilenet2_224-05.pb',input_tensor='MobilenetV2/input:0',output_tensor='MobilenetV2/Logits/AvgPool:0')
            
            #tfInference=TensorFlowInference('../DNN_models/my_tf/vgg2_nasnet224-08-0.91.pb',input_tensor='input_1:0',output_tensor='reshape_1/Mean:0',learning_phase_tensor='conv1_bn/keras_learning_phase:0')
            #tfInference=TensorFlowInference('../DNN_models/my_tf/vgg2_resnet.pb',input_tensor='input:0',output_tensor='pool5_7x7_s1:0')

        if save_video_features:
            if use_keras:
                nn_model=cnn_model
            else:
                nn_model=tfInference
            #save_video_features_to_text('../images/ijba/1n_images/video','../images/ijba/1n_images/video_my_vgg2_mobile192_features.txt',nn_model)
            save_video_features_to_text('../images/YTF/cropped','../images/YTF/vgg2_mobilenet_dnn_features1.txt',nn_model)
            
        dirs_and_files=np.array(get_files(db_dir))
        dirs=dirs_and_files[:,0]
        files=dirs_and_files[:,1]

        label_enc=preprocessing.LabelEncoder()
        label_enc.fit(dirs)
        y=label_enc.transform(dirs)
        #print ('y=',y)
        start_time = time.time()
        if use_keras:
            X=np.array([extract_keras_features(cnn_model,os.path.join(db_dir,filepath)) for filepath in files])
        elif True:
            X=tfInference.extract_features_from_images(db_dir,files)
            tfInference.close_session()
        else:
            X=np.array([tfInference.extract_features(os.path.join(db_dir,filepath)) for filepath in files])
            tfInference.close_session()
        print('--- %s seconds ---' % (time.time() - start_time))
        print ('X.shape=',X.shape)
        print ('X[0,5]=',X[:,0:6])
        np.savetxt('X.txt', X, delimiter=' ',fmt='%1.3f')
        np.savez(features_file,x=X,y=y)

        filepath='../images/lfw/my_vgg2_mobile192_all_features.txt'
        #filepath='../ijba/1n_images/my_vgg2_mobile192_features.txt'
        #filepath='../images/pubfig83/my_vgg2_mobile192_features.txt' #'vgg2_resnet_features.txt'
        #convert_features_to_text(X,dirs,filepath)
    
    data = np.load(features_file)
    X=data['x']
    X_norm=preprocessing.normalize(X,norm='l2')
    y=data['y']
    
    if False:
        dirs_and_files=get_files(db_dir)
        
        from sklearn.metrics.pairwise import pairwise_distances
        pair_dist=pairwise_distances(X_norm)
        neighbors=pair_dist.argsort(axis=1)[:,1:]
        print(pair_dist.shape,neighbors.shape)
        avg_other_ind=0
        avg_inter_min_dist,avg_intra_min_dist=0,0
        intra_min_distances=[]
        for i in range(neighbors.shape[0]):
            #if i%100==0:
            #    print(i,y[i],y[neighbors[i][0]],neighbors[i][0],pair_dist[i,neighbors[i,0]])
            for j in range(neighbors.shape[1]):
                if y[i]!=y[neighbors[i,j]]:
                    avg_other_ind+=j
                    avg_intra_min_dist+=pair_dist[i,neighbors[i,j]]
                    intra_min_distances.append(pair_dist[i,neighbors[i,j]])
                    if False and pair_dist[i,neighbors[i,j]]==0:
                        print(i,y[i],y[neighbors[i][0]],neighbors[i][0],pair_dist[i,neighbors[i,0]])
                        print(X_norm[i,:])
                        print(X_norm[neighbors[i][0],:])
                        print(dirs_and_files[i],dirs_and_files[neighbors[i][0]])
                    break
            for j in range(neighbors.shape[1]):
                if y[i]==y[neighbors[i,j]]:
                    avg_inter_min_dist+=pair_dist[i,neighbors[i,j]]
                    break
        avg_other_ind/=neighbors.shape[0]
        avg_intra_min_dist/=neighbors.shape[0]
        avg_inter_min_dist/=neighbors.shape[0]
        
        print('avg_other_ind=',avg_other_ind+1)
        print('avg_inter_min_dist=',avg_inter_min_dist)
        print('avg_intra_min_dist=',avg_intra_min_dist)
        intra_min_distances.sort()
        print('intra min=',intra_min_distances[0],' 1 percentile=',intra_min_distances[len(intra_min_distances)//100],' 5 percentile=',intra_min_distances[len(intra_min_distances)//20])
        #sys.exit(0)
    
    y_l=list(y)
    indices=[i for i,el in enumerate(y_l) if y_l.count(el) > 1]
    y=y[indices]
    label_enc=preprocessing.LabelEncoder()
    label_enc.fit(y)
    y=label_enc.transform(y)
    X_norm=X_norm[indices,:]
    print('after loading: num_classes=',len(label_enc.classes_),' X shape:',X.shape)
    if True:
        pca_components=256
        classifiers=[]
        #classifiers.append(['lightGBM',LGBMClassifier(max_depth=3,n_estimators=200)])
        #classifiers.append(['xgboost',XGBClassifier(max_depth=3,n_estimators=200)])
        classifiers.append(['k-NN',KNeighborsClassifier(n_neighbors=1,p=2)])
        classifiers.append(['k-NN+PCA',Pipeline(steps=[('pca', PCA(n_components=pca_components)), ('classifier', KNeighborsClassifier(n_neighbors=1,p=2))])])
        #classifiers.append(['k-NN chisq',KNeighborsClassifier(n_neighbors=1,metric=chi2dist)])
        #classifiers.append(['k-NN KL',KNeighborsClassifier(n_neighbors=1,metric=KL_dist)])
        #classifiers.append(['k-NN mahalonbis',KNeighborsClassifier(1,metric='mahalanobis',metric_params={'V': np.cov(X)})])
        #classifiers.append(['rf',RandomForestClassifier(n_estimators=100,max_depth=2)])
        #classifiers.append(['svm',SVC()])
        for cls_name,classifier in classifiers:
            print(cls_name)
            classifier_tester(classifier,X_norm,y)
    else:
        classifier=KNeighborsClassifier(1)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X_norm, y, test_size=test_size, random_state=42, stratify=y)
        print (X_train.shape,X_test.shape)
        print(y_train,y_test)
        classifier.fit(X_train,y_train)
        y_test_pred=classifier.predict(X_test)
        acc=100.0*(y_test==y_test_pred).sum()/len(y_test)
        print('acc=',acc)
