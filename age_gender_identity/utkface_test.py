from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import cv2
import time
import math

MODEL_DIR = 'D:/src_code/DNN_models/age_gender/'

adience_range=False
adience_age_list=[(0, 2),(4, 6),(8, 12),(15, 20),(25, 32),(38, 43),(48, 53),(60, 100)]
def get_age_range(real_age):
    for ind in range(len(adience_age_list)-1):
        if real_age<=(adience_age_list[ind][1]+adience_age_list[ind+1][0])/2:
            return ind
    return len(adience_age_list)-1

if True:
    from facial_analysis import FacialImageProcessing,is_image,age_gender_one_model
    imgProcessing=FacialImageProcessing(False)
    def get_age_female(draw):
        img=cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)
        if age_gender_one_model:
            age,gender,_=imgProcessing.age_gender_fun(img)
        else:
            age=imgProcessing.age_fun(img)
            gender=imgProcessing.gender_fun(img)
        is_female=0 if FacialImageProcessing.is_male(gender) else 1
        #print(age,gender,is_female)
        return age,is_female
elif False:
    import tensorflow as tf
    adience_range=True
    RESIZE_FINAL = 227
    src_dir=os.path.join(MODEL_DIR,'rude-carnie-master')
    
    if False: #convert models to pb
        sys.path.append(src_dir)
        from model import select_model, get_checkpoint
        from tensorflow.python.framework.graph_util import convert_variables_to_constants    
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # tf.reset_default_graph()
            model_type = 'inception'
            if False: #gender
                nlabels=2
                model_dir ='21936'
                outname='gender_net'
            else:
                nlabels=8
                model_dir ='22801'
                outname='age_net'

            model_fn = select_model('inception')


            with tf.device('/gpu:0'):
                images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])

                logits = model_fn(nlabels, images, 1, False)
                init = tf.global_variables_initializer()
                requested_step = None
                checkpoint_path = '%s' % (os.path.join(src_dir,'inception',model_dir))
                model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, 'checkpoint')
                saver = tf.train.Saver()
                saver.restore(sess, model_checkpoint_path)
                softmax_output = tf.nn.softmax(logits, name='logits')
                
            graph = sess.graph
            keep_var_names=None
            clear_devices=True
            with graph.as_default():
                freeze_var_names = list(set(v.op.name for v in tf.global_variables()))
                output_names = ['logits']
                output_names += [v.op.name for v in tf.global_variables()]
                input_graph_def = graph.as_graph_def()
                if clear_devices:
                    for node in input_graph_def.node:
                        node.device = ''
                frozen_graph = convert_variables_to_constants(sess, input_graph_def,
                                                              output_names, freeze_var_names)
                tf.train.write_graph(frozen_graph, os.path.join(src_dir,'inception'), outname+'.pb', as_text=False)
                sys.exit()
    else:
        from facial_analysis import FacialImageProcessing
        with tf.Graph().as_default() as full_graph:
            for model_file in ['gender','age']:
                tf.import_graph_def(FacialImageProcessing.load_graph_def(os.path.join(src_dir,'inception',model_file+'_net.pb')), name=model_file)
        
        sess=tf.Session(graph=full_graph)
        #print([n.name for n in full_graph.as_graph_def().node])
        gender_out=full_graph.get_tensor_by_name('gender/logits:0')
        gender_in=full_graph.get_tensor_by_name('gender/Placeholder:0')
        age_out=full_graph.get_tensor_by_name('age/logits:0')
        age_in=full_graph.get_tensor_by_name('age/Placeholder:0')
        
        def get_age_female(draw):
            img=cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(img, (RESIZE_FINAL,RESIZE_FINAL))
            x=resized_image.astype(np.float32)
            face_imgs=np.expand_dims(x, axis=0)
            predicted_genders = sess.run(gender_out, feed_dict={gender_in: face_imgs})
            predicted_ages = sess.run(age_out, feed_dict={age_in: face_imgs})
            is_female=1 if predicted_genders[0][0]<0.5 else 0
            predicted_age_range=np.argmax(predicted_ages)
            predicted_age=sum(adience_age_list[predicted_age_range])/2
            #print(predicted_genders,predicted_ages,is_female,predicted_age_range,predicted_age)
            return predicted_age,is_female
        
elif False:
    import tensorflow as tf
    multi_task_dir=os.path.join(MODEL_DIR,'multi-task-learning-master')
    sys.path.append(multi_task_dir)
    import BKNetStyle2 as BKNetStyle
    
    def load_network():
        sess = tf.Session()
        x = tf.placeholder(tf.float32, [None, 48, 48, 1])
        y_smile_conv, y_gender_conv, y_age_conv, phase_train, keep_prob = BKNetStyle.BKNetModel(x)
        print('Restore model')
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(multi_task_dir,'save/current5/model-age101.ckpt.index'))
        print('OK')
        return sess, x, y_smile_conv, y_gender_conv, y_age_conv, phase_train, keep_prob
    
    sess, x, y_smile_conv, y_gender_conv, y_age_conv, phase_train, keep_prob = load_network()
    def get_age_female(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(48, 48))
        img = (img - 128) / 255.0
        T = np.zeros([48, 48, 1])
        T[:, :, 0] = img
        test_img = []
        test_img.append(T)
        test_img = np.asarray(test_img)

        predict_y_gender_conv,predict_y_age_conv = sess.run([y_gender_conv,y_age_conv], feed_dict={x: test_img, phase_train: False, keep_prob: 1})
        predict_y_age=np.argmax(predict_y_age_conv)
        #print(predict_y_gender_conv,predict_y_age)
        is_female=1 if np.argmax(predict_y_gender_conv)==0 else 0
        return predict_y_age,is_female

elif False:
    import tensorflow as tf
    import inception_resnet_v1
    def load_network(model_path):
        sess = tf.Session()
        images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
        images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(tf.cast(frame, tf.float32)), images_pl)
        #images_norm = tf.map_fn(lambda frame: (tf.cast(frame, tf.float32) - 127.5)/128.0, images_pl)
        
        train_mode = tf.placeholder(tf.bool)
        age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8,
                                                                     phase_train=train_mode,
                                                                     weight_decay=1e-5)
        gender_prob=tf.nn.softmax(gender_logits)
        gender = tf.argmax(gender_prob, 1)
        age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
        age_prob=tf.nn.softmax(age_logits)
        age = tf.reduce_sum(tf.multiply(age_prob, age_), axis=1)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore model!")
        else:
            print("failed to restore")
        return sess,age,gender,train_mode,images_pl
        
    sess, age, gender, train_mode,images_pl = load_network(os.path.join(MODEL_DIR,'Age-Gender-Estimate-TF'))
    def get_age_female(draw):
        img=cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(img, (160,160))
        x=resized_image.astype(np.float32)
        face_imgs=np.expand_dims(x, axis=0)
        predicted_ages,predicted_genders = sess.run([age, gender], feed_dict={images_pl: face_imgs, train_mode: False})
        #print(predicted_genders,predicted_ages)
        is_female=1 if predicted_genders[0] == 0 else 0
        return predicted_ages[0],is_female
        
elif False:
    from insightface import InsightFace
    model=InsightFace()
    def get_age_female(img):
        img=model.prep_image(img)
        face_imgs=np.expand_dims(img, axis=0)
        prediction = model.predict(face_imgs)
        predicted_genders, predicted_ages = model.decode_prediction(prediction)
        
        #print(predicted_genders,predicted_ages)
        is_female=1 if predicted_genders[0] < 0.5 else 0
        return predicted_ages[0],is_female

elif False:
    agendernet_dir=os.path.join(MODEL_DIR,'Agendernet-master','model')
    sys.path.append(agendernet_dir)
    from mobilenetv2 import AgenderNetMobileNetV2
    model = AgenderNetMobileNetV2()
    model.load_weights(os.path.join(agendernet_dir,'weight/mobilenetv2/model.10-3.8290-0.8965-6.9498.h5'))

    def get_age_female(img):
        resized_image = cv2.resize(img, (model.input_size, model.input_size))
        face_imgs=np.expand_dims(resized_image, axis=0)
        face_imgs = model.prep_image(face_imgs)
        prediction = model.predict(face_imgs)
        predicted_genders, predicted_ages = model.decode_prediction(prediction)
        
        #print(predicted_genders,predicted_ages)
        is_female=1 if predicted_genders[0] < 0.5 else 0
        return predicted_ages[0],is_female

elif False:
    ssrnet_dir=os.path.join(MODEL_DIR,'SSR-Net-master','demo')
    sys.path.append(ssrnet_dir)
    from SSRNET_model import SSR_net, SSR_net_general
    img_size = 64
    
    def load_models():
        weight_file = os.path.join(ssrnet_dir,'../pre-trained/morph2/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5')
        weight_file_gender = os.path.join(ssrnet_dir,'../pre-trained/wiki_gender_models/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5')        
        # load model and weights
        stage_num = [3,3,3]
        lambda_local = 1
        lambda_d = 1
        model = SSR_net(img_size,stage_num, lambda_local, lambda_d)()
        model.load_weights(weight_file)

        model_gender = SSR_net_general(img_size,stage_num, lambda_local, lambda_d)()
        model_gender.load_weights(weight_file_gender)
        return model, model_gender
    
    model, model_gender=load_models()
    def get_age_female(img):
        resized_image = cv2.resize(img, (img_size, img_size))
        resized_image=cv2.normalize(resized_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        face_imgs=np.expand_dims(resized_image, axis=0)
        predicted_ages = model.predict(face_imgs)
        predicted_genders = model_gender.predict(face_imgs)
        
        #print(predicted_genders,predicted_ages)
        is_female=1 if predicted_genders[0] < 0.5 else 0
        return predicted_ages[0],is_female

else:
    from wide_resnet import WideResNet
    from keras.utils.data_utils import get_file
    face_size=64
    def load_model():
        model = WideResNet(face_size, depth=16, k=8)()
        #WRN_WEIGHTS_PATH = "https://github.com/Tony607/Keras_age_gender/releases/download/V1.0/weights.18-4.06.hdf5"
        WRN_WEIGHTS_PATH = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"
        fpath = get_file('weights.18-4.06.hdf5', #'weights.28-3.73.hdf5',
                         WRN_WEIGHTS_PATH,
                         cache_subdir=MODEL_DIR)
        model.load_weights(fpath)
        return model
    
    model=load_model()
    def get_age_female(img):
        resized_image = cv2.resize(img, (face_size,face_size))
        face_imgs=np.expand_dims(resized_image, axis=0)
        results = model.predict(face_imgs)
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()
        #print(predicted_genders,predicted_ages)
        is_female=1 if predicted_genders[0][0] > 0.5 else 0
        return predicted_ages[0],is_female

import csv
def get_files_from_csv(db_dir):
    files=[]
    with open(os.path.join(db_dir,'utk_test.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                #print(line_count,row[1])
                files.append(row[1])
            line_count+=1
    return files
                
from random import shuffle
def process_utkface(db_dir):
    all_set=True
    if all_set:
        files=[f for f in next(os.walk(db_dir))[2] if f.lower().endswith('jpg')]
    else:
        files=get_files_from_csv(db_dir)
    shuffle(files)
    
    gender_acc=0
    age_acc=0
    age_delta=5
    age_mae=0
    num_files=len(files)
    t = time.time()        
    for f in files[:num_files]:
        fields=f.split('_')
        real_age,real_is_female=int(fields[0]),int(fields[1])
        #print(f,real_age,real_is_female)
        draw = cv2.imread(os.path.join(db_dir,f))
        age,is_female=get_age_female(draw)
        if not all_set:
            if age<21:
                age=21
            elif age>60:
                age=60

        if is_female==real_is_female:
            gender_acc+=1

        if adience_range:
            real_age_range=get_age_range(real_age)
            age_range=get_age_range(age)
            #print(real_age_range,age_range)
            if real_age_range==age_range:
                age_acc+=1
        else:
            if abs(age-real_age)<=age_delta:
                age_acc+=1
            age_mae+=abs(age-real_age)

    elapsed = time.time() - t

    print('num_files=',num_files,' elapsed=',elapsed,' gender accuracy=',gender_acc/num_files,' age accuracy=',age_acc/num_files, ' age MAE=',age_mae/num_files)

if __name__ == '__main__':
    process_utkface('D:/datasets/UTKFace')