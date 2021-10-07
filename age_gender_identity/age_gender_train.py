from __future__ import print_function
import argparse
import sys
import os.path
import os
from os import listdir
from os.path import isfile, join
import datetime


#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNet
from keras.engine.topology import Input
from keras.layers import Flatten, Dense, Dropout,GlobalAveragePooling2D, Reshape, Conv2D, Activation, MaxPooling2D, Multiply, BatchNormalization, Lambda
from keras.models import Model, Sequential
from keras.optimizers import Adam, Nadam
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras.applications.imagenet_utils import preprocess_input
from keras.regularizers import l2
from keras.losses import *
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras import backend as K

#from nasnet import NASNetMobile

import numpy as np
np.random.seed(123) 

activ_func,loss_func,class_mode,metric,monitor='sigmoid',binary_crossentropy,'binary','accuracy','val_acc'
#activ_func,loss_func,class_mode,metric,monitor='linear',mean_absolute_error,'sparse','mse','val_loss'
#activ_func,loss_func,class_mode,metric,monitor='softmax',categorical_crossentropy,'categorical','accuracy','val_acc'


use_mobilenet=True
input_height=input_width=192 #224 #128 #192 #224 #200

def preprocess(x,**kwargs):
    x_new = preprocess_input(x)
    return x_new


def get_net():
    if use_mobilenet:
        return MobileNet(include_top=False, weights=None,input_shape=(input_height, input_width, 3))
    else:
        return NASNetMobile(include_top=False, weights='imagenet',input_shape=(input_height, input_width, 3))

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

def load_model(modelpath):
    base_model = get_net()    
    last_model_layer = base_model.output

    if use_mobilenet:
        x = GlobalAveragePooling2D()(last_model_layer)
        out = Reshape((1,1,1024), name='reshape_1')(x)
    else:
        out = GlobalAveragePooling2D(name='reshape_1')(last_model_layer)
    cnn_model = Model(base_model.input, out)
    cnn_model.load_weights(modelpath)
    return cnn_model,base_model


def convert_to_tf(modelpath):
    K.set_learning_phase(0)
    config = tf.ConfigProto(device_count={'CPU':1,'GPU':0})
    KTF.set_session(tf.Session(config=config))
    
    filename,ext=os.path.splitext(os.path.basename(modelpath))
    print (filename, ext)
    base_model = get_net()        
    last_model_layer = base_model.output
    x = GlobalAveragePooling2D(name='global_pooling')(last_model_layer)
    
    x = Dropout(0.5)(x)
    
    x=Dense(256, activation='relu', kernel_regularizer=l2(4e-5), name='feats')(x)
    x = Dropout(0.2)(x)
    age_preds = Dense(100, activation='softmax', kernel_regularizer=l2(4e-5), name='age_pred')(x)
    gender_preds = Dense(1, activation='sigmoid', kernel_regularizer=l2(4e-5), name='gender_pred')(x)

    summary_model= Model(base_model.input,[age_preds,gender_preds])

    summary_model.load_weights(modelpath)
    summary_model.summary()

    outputs=[o.op.name for o in summary_model.output]
    print (outputs)
    frozen_graph = freeze_session(K.get_session(), output_names=outputs)
    tf.train.write_graph(frozen_graph, 'models', filename+'.pb', as_text=False)


if __name__ == '__main__':
    #config = tf.ConfigProto()
    #config.gpu_options.allocator_type = 'BFC'
    #KTF.set_session(tf.Session(config=config))

    parser = argparse.ArgumentParser(description="Training script for age/gender recognition.")

    parser.add_argument("--datapath", type=str, default='/home/datasets/images/DEX_IMDB_faces/tf2', help="Path where image data is stored.")
    parser.add_argument("--modelpath", type=str, default='', help="Path where trained model is stored.")
    
    args = parser.parse_args()
    if args.modelpath!='':
        convert_to_tf(args.modelpath)
        sys.exit()
   
    batch_size = 80
 
    train_datagen = ImageDataGenerator(shear_range=0.3, #0.2
                                       rotation_range=10,
                                       zoom_range=0.2, #0.1
                                       width_shift_range=0.1,height_shift_range=0.1,
                                       preprocessing_function=preprocess,
                                       horizontal_flip=True)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess)
    
    target_size=(input_height, input_width)


    age_train_dir,age_val_dir=args.datapath + '/age/train/',args.datapath + '/age/val/'
    gender_train_dir,gender_val_dir=args.datapath + '/gender/train/',args.datapath + '/gender/val/'    

    age_train_generator = train_datagen.flow_from_directory(
        age_train_dir, target_size=target_size,
        batch_size=batch_size, class_mode='categorical')

    age_val_generator = test_datagen.flow_from_directory(
        age_val_dir, target_size=target_size,
        batch_size=batch_size, class_mode='categorical')

    print(age_train_generator.class_indices)
    print(age_val_generator.class_indices)

    gender_train_generator = train_datagen.flow_from_directory(
        gender_train_dir, target_size=target_size,
        batch_size=batch_size, class_mode='binary')

    gender_val_generator = test_datagen.flow_from_directory(
        gender_val_dir, target_size=target_size,
        batch_size=batch_size, class_mode='binary')

    age_train_samples,age_classes_num=age_train_generator.samples,age_train_generator.num_classes
    age_val_samples=age_val_generator.samples
    print('age after read',age_train_samples,age_val_samples,age_classes_num)
    
    gender_train_samples,gender_classes_num=gender_train_generator.samples,gender_train_generator.num_classes
    gender_val_samples=gender_val_generator.samples
    print('gender after read',gender_train_samples,gender_val_samples,gender_classes_num)


    model_file='../models/vgg2_mobilenet.h5'
    print ('loading model %s'%model_file)
    _,base_model = load_model(model_file)

    last_model_layer = base_model.output
    x = GlobalAveragePooling2D(name='global_pooling')(last_model_layer)

    x = Dropout(0.5)(x)
    x=Dense(256, activation='relu', kernel_regularizer=l2(4e-5), name='feats')(x)
    x = Dropout(0.5)(x)
    age_preds = Dense(age_classes_num, activation='softmax', kernel_regularizer=l2(4e-5), name='age_pred')(x)
    gender_preds = Dense(1, activation='sigmoid', kernel_regularizer=l2(4e-5), name='gender_pred')(x)

    age_model = Model(base_model.input,age_preds)
    gender_model = Model(base_model.input,gender_preds)
    summary_model= Model(base_model.input,[age_preds,gender_preds])



    age_steps_per_epoch=age_train_samples // batch_size
    gender_steps_per_epoch=gender_train_samples // batch_size
    print('steps per epoch',age_steps_per_epoch,gender_steps_per_epoch)
    max_steps=max(age_steps_per_epoch,gender_steps_per_epoch)
    
    def train_age_gender(num_epochs, filepath=None):
        age_val_acc,gender_val_acc=0.0,0.0
        prev_age_val_acc,prev_gender_val_acc=0.0,0.0

        for iepoch in range(num_epochs):
            print ('\nEpoch:%d of %d'%(iepoch+1,num_epochs))
            age_loss,age_acc=0.0,0.0
            gender_loss,gender_acc=0.0,0.0
            cur_age_batch,cur_gender_batch=0,0
            for ibatch in range(max_steps):
                if ibatch>=(max_steps-age_steps_per_epoch):
                    age_batch_x,age_batch_y=age_train_generator.next()
                    #print('age batch:',age_batch_x.shape,age_batch_y.shape)
                    l,a=age_model.train_on_batch(age_batch_x,age_batch_y)
                    age_loss=(age_loss*cur_age_batch+l)/(cur_age_batch+1)
                    age_acc=(age_acc*cur_age_batch+a)/(cur_age_batch+1)
                    cur_age_batch+=1
                
                if ibatch>=(max_steps-gender_steps_per_epoch):
                    gender_batch_x,gender_batch_y=gender_train_generator.next()
                    #print('gender batch:',gender_batch_x.shape,gender_batch_y.shape)
                    l,a=gender_model.train_on_batch(gender_batch_x,gender_batch_y)
                    gender_loss=(gender_loss*cur_gender_batch+l)/(cur_gender_batch+1)
                    gender_acc=(gender_acc*cur_gender_batch+a)/(cur_gender_batch+1)
                    cur_gender_batch+=1
                print ('%d/%d age_loss=%.4f age_acc=%.4f gender_loss=%.4f gender_acc=%.4f\r'%(ibatch+1,max_steps,age_loss,age_acc,gender_loss,gender_acc),end='')
                sys.stdout.flush()
            age_val_loss,age_val_acc=age_model.evaluate_generator(age_val_generator, age_val_samples//batch_size)
            print('\nage val: %.4f %.4f'%(age_val_loss,age_val_acc))
            gender_val_loss,gender_val_acc=gender_model.evaluate_generator(gender_val_generator, gender_val_samples//batch_size)
            print('gender val: %.4f %.4f'%(gender_val_loss,gender_val_acc))
            if filepath!=None and (prev_age_val_acc<age_val_acc or prev_gender_val_acc<gender_val_acc):
                summary_model.save(filepath%(iepoch+1,age_val_acc,gender_val_acc))
            if prev_age_val_acc<age_val_acc:
                prev_age_val_acc=age_val_acc
            if prev_gender_val_acc<gender_val_acc:
                prev_gender_val_acc=gender_val_acc
        
        return age_val_acc,gender_val_acc


    filepath='../models/age_gender_tf2_224_deep_fn'
    filepath+='-%02d-%.2f-%.2f.hdf5'
    #filepath+='-{epoch:02d}.hdf5' 

    saved_model='../models/age_gender_tf2_224_deep-03-0.13-0.97.hdf5'
    if not os.path.exists(saved_model):
        for l in base_model.layers:
            l.trainable=False

        opt=Adam(lr=1e-3,decay=1e-6)
        #opt=Nadam()
        age_model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
        gender_model.compile(loss='binary_crossentropy',optimizer=opt, metrics=['accuracy'])
        summary_model.summary()

        age_val_acc,gender_val_acc=train_age_gender(3,filepath)

        #summary_model.save(filepath%(0,age_val_acc,gender_val_acc))
        sys.exit(0)

        for l in base_model.layers:
            l.trainable=True

    else:
        summary_model.load_weights(saved_model)
    
    

    opt=Adam(lr=1e-4,decay=1e-6)
    age_model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
    gender_model.compile(loss='binary_crossentropy',optimizer=opt, metrics=['accuracy'])
    summary_model.summary()
    
    
    train_age_gender(30,filepath)
