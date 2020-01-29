from __future__ import print_function
import argparse
import sys
import os.path
import os
from os import listdir
from os.path import isfile, join
import datetime


#os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import mobilenet,mobilenet_v2,densenet,inception_resnet_v2,inception_v3
from keras.layers import Flatten, Dense, Dropout,GlobalAveragePooling2D, Reshape, Conv2D, Activation, MaxPooling2D, BatchNormalization, Lambda
from keras.models import Model, Sequential,model_from_json
from keras.optimizers import Adam, Nadam
from keras.callbacks import TensorBoard,ModelCheckpoint,EarlyStopping
from keras.applications.imagenet_utils import preprocess_input
from keras.regularizers import l2
from keras.losses import *
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras import backend as K

import numpy as np
np.random.seed(123) 

activ_func,loss_func,class_mode,metric,monitor='softmax',categorical_crossentropy,'categorical','accuracy','val_acc'

input_height=input_width=224 #192 #128 #192 #224 #200


def count_of_files(dir):
    #print 'dir=',dir
    dirs = next(os.walk(dir))[1]
    file_count=0
    for d in dirs:
        #file_count+=len(next(os.walk(os.path.join(dir,d)))[2])
        directory=os.path.join(dir,d)
        file_count+=sum(1 for entry in listdir(directory) if isfile(join(directory,entry)))
    return file_count,len(dirs)

#net_model=mobilenet_v2
net_model=mobilenet
def get_net():
    if False:
        return net_model.MobileNetV2(alpha=1.4,include_top=False, weights='imagenet',input_shape=(input_height, input_width, 3))
    else:
        return net_model.MobileNet(include_top=False, weights='imagenet',input_shape=(input_height, input_width, 3))

def get_predictive_model(base_model,classes_num):
    last_model_layer = base_model.output
    x = GlobalAveragePooling2D(name='reshape_1')(last_model_layer)
    preds = Dense(classes_num, activation=activ_func, kernel_regularizer=l2(4e-5), name='predictions')(x)
    f_model = Model(base_model.input, preds)
    return f_model

def model_architecture(classes_num):
    #net_model = VGG16(weights='imagenet', include_top=False,
    #                     input_shape=(input_height, input_width, 3))
    #net_model = ResNet50(weights='imagenet', include_top=False,
    #                      input_shape=(input_height, input_width, 3))
    # net_model = Xception(weights='imagenet', include_top=False, input_shape=(input_height, input_width, 3))
    # net_model = VGG19(weights='imagenet', include_top=False,
    #                      input_shape=(input_height, input_width, 3))
    base_model = get_net()    
    return get_predictive_model(base_model,classes_num),base_model

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

    out = GlobalAveragePooling2D(name='reshape_1')(last_model_layer)
    cnn_model = Model(base_model.input, out)
    cnn_model.load_weights(modelpath)
    return cnn_model,base_model

from tensorflow.python.framework.graph_util import convert_variables_to_constants
def convert_to_tf(modelpath):
    K.set_learning_phase(0)
    config = tf.ConfigProto(device_count={'CPU':1,'GPU':0})
    KTF.set_session(tf.Session(config=config))
    filename,ext=os.path.splitext(os.path.basename(modelpath))
    print (filename, ext)
    if ext=='.hdf5':
        model,_ = model_architecture(9131)
        model.summary()
        model.load_weights(modelpath)
        out = model.get_layer('reshape_1').output
        print('out=',out)
        
        cnn_model = Model(model.input, out)
        #cnn_model.save('models/'+filename+'.h5')

        cnn_model.save_weights('weights.h5')
        model_json = cnn_model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)

        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights('weights.h5')

        loaded_model.save('models/'+filename+'.h5')
    else:
        saved_model = load_model(modelpath)[0]
        output_names=[saved_model.output.op.name]
        print (output_names)

        session=K.get_session()
        graph = session.graph
        keep_var_names=None
        clear_devices=True
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
        tf.train.write_graph(frozen_graph, 'models', filename+'.pb', as_text=False)

if __name__ == '__main__':
    #config = tf.ConfigProto()
    #config.gpu_options.allocator_type = 'BFC'
    #KTF.set_session(tf.Session(config=config))

    parser = argparse.ArgumentParser(description="Training script for face recognition.")

    parser.add_argument("--datapath", type=str, default='/home/datasets/images/vgg_face_dataset/vggface-2/tf1', help="Path where image data is stored.")
    parser.add_argument("--modelpath", type=str, default='', help="Path where trained model is stored.")
    
    train_batch_size=val_batch_size= 32 #76 # 88 #104 #128
    
    args = parser.parse_args()
    if args.modelpath!='':
        convert_to_tf(args.modelpath)
        sys.exit()

    train_path=args.datapath + '/train/'
    val_path=args.datapath + '/val/'

    train_datagen = ImageDataGenerator(shear_range=0.3, #0.2
                                       rotation_range=10,
                                       zoom_range=0.2, #0.1
                                       width_shift_range=0.1,height_shift_range=0.1,
                                       horizontal_flip=True,preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    target_size=(input_height, input_width)

    test_generator = test_datagen.flow_from_directory(
        val_path,
        target_size=target_size,
        batch_size=val_batch_size, class_mode=class_mode)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=target_size,
        batch_size=train_batch_size, class_mode=class_mode)

    nb_train_samples=train_generator.samples
    classes_num=train_generator.num_classes
    nb_validation_samples=test_generator.samples
    print('after read',nb_train_samples,nb_validation_samples,classes_num)

    
    model_file=''
    model,base_model = model_architecture(classes_num)    

    opt=Adam(lr=1e-3,decay=1e-5)
    model.compile(loss=loss_func, optimizer=opt, metrics=[metric])
    model.summary()
    
    #filepath="models/casia_mobilenet128"
    filepath="models/vgg2_mobilenet2_224"
    #filepath="models/gender_mobilenet192_tf15"
    #filepath='models/age_mobilenet192_tf1_regr'
    #filepath='models/age_simple_tf2'
    #saved_weights_filepath=filepath+'-03-0.64.hdf5'
    #model.load_weights(saved_weights_filepath)


    last_path='-{epoch:02d}-{val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath+last_path, monitor=monitor, verbose=1, save_best_only=True, mode='auto')
    es=EarlyStopping(monitor='val_acc',patience=2)
    callbacks=[checkpoint,es]
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // train_batch_size,
        epochs=16,
        validation_data=test_generator,
        validation_steps=nb_validation_samples // val_batch_size,
        callbacks=callbacks)
    model.save(filepath+'.hdf5')
