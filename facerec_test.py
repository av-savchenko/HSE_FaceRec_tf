import argparse
import sys
import os.path
import os
import math
import datetime, time
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from scipy import misc
#from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from PIL import Image


import numpy as np
np.random.seed(123)  # for reproducibility

use_my_cnn=False
use_keras=False

DATASET_PATH='D:/datasets/lfw_ytf/lfw'#_faces'


img_extensions=['.jpg','.jpeg','.png']
def is_image(path):
    _, file_extension = os.path.splitext(path)
    return file_extension.lower() in img_extensions

def get_files(db_dir):
    return [[d,os.path.join(d,f)] for d in next(os.walk(db_dir))[1] for f in next(os.walk(os.path.join(db_dir,d)))[2] if is_image(f)]

def load_graph(frozen_graph_filename, prefix=''):
    with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name=prefix)
    return graph

class TensorFlowInference:
    def __init__(self,frozen_graph_filename,input_tensor,output_tensor,learning_phase_tensor=None, convert2BGR=True, imageNetUtilsMean=True,additional_input_value=0):
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
        if self.tf_input_image.shape.dims is None:
            w=h=160
        else:
            _,w,h,_=self.tf_input_image.shape
        self.w,self.h=int(w),int(h)
        print ('input w,h',self.w,self.h,' output shape:',self.tf_output_features.shape)

        self.convert2BGR=convert2BGR
        self.imageNetUtilsMean=imageNetUtilsMean
        self.additional_input_value=additional_input_value

    def preprocess_image(self,img_filepath,crop_center):
        if crop_center:
            orig_w,orig_h=250,250
            img = misc.imread(img_filepath, mode='RGB')
            img = misc.imresize(img, (orig_w,orig_h), interp='bilinear')
            w1,h1=128,128
            dw=(orig_w-w1)//2
            dh=(orig_h-h1)//2
            box = (dw, dh, orig_w-dw, orig_h-dh)
            img=img[dh:-dh,dw:-dw]
        else:
            img = misc.imread(img_filepath, mode='RGB')
        
        x = misc.imresize(img, (self.w,self.h), interp='bilinear').astype(float)
        
        if self.convert2BGR:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
            # Zero-center by mean pixel
            if self.imageNetUtilsMean: #imagenet.utils caffe
                x[..., 0] -= 103.939
                x[..., 1] -= 116.779
                x[..., 2] -= 123.68
            else: #vggface-2
                x[..., 0] -= 91.4953
                x[..., 1] -= 103.8827
                x[..., 2] -= 131.0912
        else:
            x=(x-127.5)/128.0
        return x
        
    def extract_features(self,img_filepath,crop_center=False):
        x=self.preprocess_image(img_filepath,crop_center)
        x = np.expand_dims(x, axis=0)
        feed_dict={self.tf_input_image: x}
        if self.tf_learning_phase is not None:
            feed_dict[self.tf_learning_phase]=self.additional_input_value
        preds = self.tf_sess.run(self.tf_output_features, feed_dict=feed_dict).reshape(-1)
        return preds
    
    def close_session(self):
        self.tf_sess.close()


def extract_keras_features(model,img_filepath):
    _,w,h,_=model.input.shape
    w,h=int(w),int(h)
    if False:
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

def chi2dist(x, y):
    sum=x+y
    chi2array=np.where(sum>0, (x-y)**2/sum, 0)
    return np.sum(chi2array)

def KL_dist(x, y):
    KL_array=(x+0.001)*np.log((x+0.001)/(y+0.001))
    return np.sum(KL_array)

def get_single_image_per_class_cv(y, n_splits=10,random_state=0):
    res_cv=[]
    inds = np.arange(len(y))
    np.random.seed(random_state)
    for _ in range(n_splits):
        inds_train, inds_test = [], []

        for lbl in np.unique(y):
            tmp_inds = inds[y == lbl]
            np.random.shuffle(tmp_inds)
            last_ind=1
            #last_ind=math.ceil(len(tmp_inds)/2)
            if last_ind==0 and len(tmp_inds)>0:
                last_ind=1
            inds_train.extend(tmp_inds[:last_ind])
            inds_test.extend(tmp_inds[last_ind:])
            
        inds_train = np.array(inds_train)
        inds_test = np.array(inds_test)
    
        res_cv.append((inds_train, inds_test))
    return res_cv

def classifier_tester(classifier,x,y):
    sss=get_single_image_per_class_cv(y)
    #sss=model_selection.StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
    scores=model_selection.cross_validate(classifier,x, y, scoring='accuracy',cv=sss)
    acc=scores['test_score']
    print('accuracies=',acc*100)
    print('total acc=',round(acc.mean()*100,2),round(acc.std()*100,2))
    print('test time=',scores['score_time'])

if __name__ == '__main__':
    crop_center=True
    #features_file='lfw_ytf_mobilenet_vgg2_features.npz'
    #features_file='lfw_ytf_vgg2resnet_features.npz'
    #features_file='lfw_ytf_facenet_inception_resnet_features.npz'
    features_file='lfw_ytf_insightface_features.npz'
    
    save_video_features=False
    if not os.path.exists(features_file) or save_video_features:
        os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

        if use_keras:
            from keras.engine import  Model
            from keras.preprocessing import image
            from keras import backend as K
            K.set_learning_phase(0)

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
                
                model = Model(net_model.input, x)
                model.load_weights('models/vgg2_mobilenet.h5')
                out = model.get_layer('reshape_1').output
                print('out=',out)
            else:
                from keras_vggface.vggface import VGGFace
                from keras_vggface.utils import preprocess_input
                model_name, layer='vgg16','fc7/relu'
                #model_name, layer='resnet50','avg_pool'
                model = VGGFace(model=model_name) # pooling: None, avg or max
                out = model.get_layer(layer).output
            
            cnn_model = Model(model.input, out)
            cnn_model.summary()
        else:
            import tensorflow as tf            
            #tfInference=TensorFlowInference('age_gender_identity/age_gender_tf2_new-01-0.14-0.92.pb',input_tensor='input_1:0',output_tensor='global_pooling/Mean:0',convert2BGR=True, imageNetUtilsMean=True)
            #tfInference=TensorFlowInference('models/vgg2_mobilenet.pb',input_tensor='input_1:0',output_tensor='reshape_1/Reshape:0',learning_phase_tensor='conv1_bn/keras_learning_phase:0',convert2BGR=True, imageNetUtilsMean=True)
            #tfInference=TensorFlowInference('models/vgg2_resnet.pb',input_tensor='input:0',output_tensor='pool5_7x7_s1:0',convert2BGR=True, imageNetUtilsMean=False)
            #tfInference=TensorFlowInference('../DNN_models/facenet_inceptionresnet/20180402-114759.pb',input_tensor='input:0',output_tensor='embeddings:0',learning_phase_tensor='phase_train:0',convert2BGR=False)
            tfInference=TensorFlowInference('../DNN_models/insightface/insightface.pb',input_tensor='img_inputs:0',output_tensor='resnet_v1_50/E_BN2/Identity:0',learning_phase_tensor='dropout_rate:0',convert2BGR=False,additional_input_value=0.9)


        if False:
            dirs_and_files=np.array(get_files(DATASET_PATH))
        else: #LFW and YTF concatenation
            subjects = (line.rstrip('\n') for line in open('lfw_ytf_classes.txt'))
            dirs_and_files=np.array([[d,os.path.join(d,f)] for d in subjects for f in next(os.walk(os.path.join(DATASET_PATH,d)))[2] if is_image(f)])
            
        dirs=dirs_and_files[:,0]
        files=dirs_and_files[:,1]

        label_enc=preprocessing.LabelEncoder()
        label_enc.fit(dirs)
        y=label_enc.transform(dirs)
        start_time = time.time()
        if use_keras:
            X=np.array([extract_keras_features(cnn_model,os.path.join(DATASET_PATH,filepath)) for filepath in files])
        else:
            X=np.array([tfInference.extract_features(os.path.join(DATASET_PATH,filepath),crop_center=crop_center) for filepath in files])
            tfInference.close_session()
        print('--- %s seconds ---' % (time.time() - start_time))
        print ('X.shape=',X.shape)
        print ('X[0,5]=',X[:,0:6])
        np.savez(features_file,x=X,y=y)

    data = np.load(features_file)
    X=data['x']
    X_norm=preprocessing.normalize(X,norm='l2')
    y=data['y']
    
    y_l=list(y)
    indices=[i for i,el in enumerate(y_l) if y_l.count(el) > 1]
    y=y[indices]
    label_enc=preprocessing.LabelEncoder()
    label_enc.fit(y)
    y=label_enc.transform(y)
    X_norm=X_norm[indices,:]
    print('after loading: num_classes=',len(label_enc.classes_),' X shape:',X.shape,' X_norm shape:',X_norm.shape)
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
        #classifiers.append(['rf',RandomForestClassifier(n_estimators=100,max_depth=10)])
        #classifiers.append(['svm',SVC()])
        #classifiers.append(['linear svm',LinearSVC()])
        for cls_name,classifier in classifiers:
            print(cls_name)
            classifier_tester(classifier,X_norm,y)
    else:
        classifier=KNeighborsClassifier(1)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X_norm, y, test_size=0.5, random_state=42, stratify=y)
        print (X_train.shape,X_test.shape)
        print(y_train.shape,y_test.shape)
        print('train classes:',len(np.unique(y_train)))
        classifier.fit(X_train,y_train)
        y_test_pred=classifier.predict(X_test)
        acc=100.0*(y_test==y_test_pred).sum()/len(y_test)
        print('acc=',acc)
