from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import tensorflow as tf
import numpy as np
import cv2
import time

import subprocess, re 


def is_specialfile(path,exts):
    _, file_extension = os.path.splitext(path)
    return file_extension.lower() in exts

img_extensions=['.jpg','.jpeg','.png']
def is_image(path):
    return is_specialfile(path,img_extensions)

video_extensions=['.mov','.avi']
def is_video(path):
    return is_specialfile(path,video_extensions)
    
age_gender_one_model=True
class FacialImageProcessing:
    # minsize: minimum of faces' size
    def __init__(self, print_stat=False, mtcnn_detector=True, minsize = 32):
        self.mtcnn_detector=mtcnn_detector
        self.print_stat=print_stat
        self.minsize=minsize
        
        models_path,_ = os.path.split(os.path.realpath(__file__))
        model_files={os.path.join(models_path,'mtcnn.pb'):''}
        if age_gender_one_model:
            model_files[os.path.join(models_path,'age_gender_tf2_new-01-0.14-0.92.pb')]=''
        else:
            model_files['D:/src_code/DNN_models/my_tf/gender_mobilenet224_02-0.92.pb']='gender'
            #model_files['D:/src_code/DNN_models/my_tf/age_mobilenet192_tf1_regr.pb']='age'
            model_files['D:/src_code/DNN_models/my_tf/age_mobilenet192_tf1_softmax-13-0.15.pb']='age'
        
        with tf.Graph().as_default() as full_graph:
            for model_file in model_files:
                tf.import_graph_def(FacialImageProcessing.load_graph_def(model_file), name=model_files[model_file])
        self.sess=tf.Session(graph=full_graph)#,config=tf.ConfigProto(device_count={'CPU':1,'GPU':0}))
        if self.mtcnn_detector:
            self.pnet, self.rnet, self.onet = FacialImageProcessing.load_mtcnn(self.sess,full_graph)     
            print('MTCNN detector loaded')
        else:
            self.face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
            print('opencv detector loaded')

        if age_gender_one_model:
            self.age_gender_fun=self.load_age_gender(self.sess,full_graph)
        else:
            self.gender_fun=self.load_gender(self.sess,full_graph)
            #self.age_fun=self.load_age(self.sess,full_graph)
            self.age_fun=self.load_age(self.sess,full_graph)

    def close(self):
        self.sess.close()
    
    @staticmethod
    def is_male(gender_preds):
        return (gender_preds>=0.6)
        #return gender_preds[1]>0.5
        
    def load_age_gender(self,sess,graph):
        age_out=graph.get_tensor_by_name('age_pred/Softmax:0')
        gender_out=graph.get_tensor_by_name('gender_pred/Sigmoid:0')
        facial_features_out=graph.get_tensor_by_name('global_pooling/Mean:0')
        
        print(age_out,gender_out,facial_features_out)
        in_img=graph.get_tensor_by_name('input_1:0')
        _,w,h,_=in_img.shape
        #w=h=224
        print(in_img,w,h)
        def age_gender_fun(img):
            if True:
                resized_image = cv2.resize(img, (w,h))
            else:
                if img.shape[0]<128 or img.shape[1]<128:
                    resized_image = cv2.resize(img, (128,128))
                else:
                    resized_image = img
            # Zero-center by mean pixel
            x=resized_image.astype(np.float32)
            # 'RGB'->'BGR'
            x = x[..., ::-1]
            x[..., 0] -= 103.939
            x[..., 1] -= 116.779
            x[..., 2] -= 123.68
            x = np.expand_dims(x, axis=0)
            age_preds,gender_preds,facial_features = sess.run([age_out,gender_out,facial_features_out], feed_dict={in_img: x})
            age_preds,gender_preds = age_preds[0],gender_preds[0]
            facial_features=facial_features[0]
            
            min_age=1
            #res_age=min_age
            #for age in range(0,101-min_age):
            #    res_age+=age*age_preds[age]
            #print('res_age',res_age)
            #return res_age
            indices=age_preds.argsort()[::-1][:2]
            norm_preds=age_preds[indices]/np.sum(age_preds[indices])
            
            res_age=min_age
            for age,probab in zip(indices,norm_preds):
                res_age+=age*probab
            if self.print_stat:
                print ('gender',gender_preds)
                print ('age',res_age)
                print (indices,age_preds[indices],norm_preds)
            return res_age,gender_preds,facial_features
        return age_gender_fun

    def load_gender(self,sess,graph):
        gender_out=graph.get_tensor_by_name('gender/predictions/Sigmoid:0')
        #gender_out=graph.get_tensor_by_name('gender/prob:0')
        print(gender_out)
        print([n.name for n in graph.as_graph_def().node if 'data' in n.name])
        gender_in=graph.get_tensor_by_name('gender/input_1:0')
        #gender_in=graph.get_tensor_by_name('gender/data:0')
        _,w,h,_=gender_in.shape
        print(gender_in,w,h)
        def gender_fun(img):
            resized_image = cv2.resize(img, (w,h))
            # Zero-center by mean pixel
            x=resized_image.astype(np.float32)
            # 'RGB'->'BGR'
            x = x[..., ::-1]
            x[..., 0] -= 103.939
            x[..., 1] -= 116.779
            x[..., 2] -= 123.68
            x = np.expand_dims(x, axis=0)
            preds = sess.run(gender_out, feed_dict={gender_in: x}).reshape(-1)
            if self.print_stat:
                print ('gender',preds)
            return preds
        return gender_fun

    def load_age(self,sess,graph):
        #print([n.name for n in graph.as_graph_def().node if 'predictions' in n.name])
        #age_out=graph.get_tensor_by_name('age/predictions/BiasAdd:0')
        age_out=graph.get_tensor_by_name('age/predictions/Softmax:0')
        #age_out=graph.get_tensor_by_name('age/prob:0')
        print(age_out)
        age_in=graph.get_tensor_by_name('age/input_1:0')
        #age_in=graph.get_tensor_by_name('age/lambda_1_input:0')
        #age_in=graph.get_tensor_by_name('age/data:0')
        _,w,h,_=age_in.shape
        print(age_in,w,h)
        def age_fun(img):
            resized_image = cv2.resize(img, (w,h))
            #cv2.imwrite('face.jpg',cv2.cvtColor(resized_image,cv2.COLOR_RGB2BGR))
            # Zero-center by mean pixel
            x=resized_image.astype(np.float32)
            # 'RGB'->'BGR'
            x = x[..., ::-1]
            x[..., 0] -= 103.939
            x[..., 1] -= 116.779
            x[..., 2] -= 123.68
            x = np.expand_dims(x, axis=0)
            preds = sess.run(age_out, feed_dict={age_in: x}).reshape(-1)
            #print (preds.shape)
            #return preds
            min_age=1
            #res_age=min_age
            #for age in range(0,101-min_age):
            #    res_age+=age*preds[age]
            #return res_age
            indices=preds.argsort()[::-1][:2]
            norm_preds=preds[indices]/np.sum(preds[indices])
            
            res_age=min_age
            for age,probab in zip(indices,norm_preds):
                res_age+=age*probab

            if self.print_stat:
                print('res_age',res_age)
                print (indices)
                print (preds[indices],norm_preds)
            return res_age
            
            #return np.argmax(preds)+min_age
        return age_fun
    
    def detect_faces(self,img):
        if self.mtcnn_detector:
            bounding_boxes, points = self.mtcnn_detect_faces(img)
        else:
            bounding_boxes = []
            points=[]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1,3)#,minSize=(self.minsize,self.minsize))
            for (x,y,w,h) in faces:
                #roi_gray = gray[y:y+h, x:x+w]
                #eyes = self.eye_cascade.detectMultiScale(roi_gray)
                #if len(eyes)>0:
                bounding_boxes.append([x,y,x+w,y+h])
        return bounding_boxes, points
    
    def process_image(self,draw):
        img=cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)
        t = time.time()
        bounding_boxes, points = self.detect_faces(img)
        elapsed = time.time() - t
        if self.print_stat:
            print('detection elapsed',elapsed)
        ages,genders,facial_features,bboxes=[],[],[],[]
        for b in bounding_boxes:
            b=[int(bi) for bi in b]
            #print(b,img.shape)
            x1,y1,x2,y2=b[0:4]
            if x2>x1 and y2>y1:
                img_h,img_w,_=img.shape
                w,h=x2-x1,y2-y1
                dw,dh=10,10 #max(w//8,10),max(h//8,10) #w//6,h//6
                #sz=max(w+2*dw,h+2*dh)
                #dw,dh=(sz-w)//2,(sz-h)//2
                x1,x2=x1-dw,x2+dw
                y1,y2=y1-dh,y2+dh
                
                boxes=[[x1,y1,x2,y2]]
                
                if False: #oversampling
                    delta=10
                    boxes.append([x1-delta,y1-delta,x2-delta,y2-delta])
                    boxes.append([x1-delta,y1+delta,x2-delta,y2+delta])
                    boxes.append([x1+delta,y1-delta,x2+delta,y2-delta])
                    boxes.append([x1+delta,y1+delta,x2+delta,y2+delta])

                for ind in range(len(boxes)):
                    if boxes[ind][0]<0:
                        boxes[ind][0]=0
                    if boxes[ind][2]>img_w:
                        boxes[ind][2]=img_w
                    if boxes[ind][1]<0:
                        boxes[ind][1]=0
                    if boxes[ind][3]>img_h:
                        boxes[ind][3]=img_h
                
                avg_age,avg_gender=0,0
                for (x1,y1,x2,y2) in boxes[::-1]:
                    face_img=img[y1:y2,x1:x2,:]
                    #cv2.imwrite('face.jpg',cv2.cvtColor(face_img,cv2.COLOR_RGB2BGR))
                    t = time.time()
                    if age_gender_one_model:
                        age,gender,features=self.age_gender_fun(face_img)
                        elapsed = time.time() - t
                        if self.print_stat:
                            print('age gender elapsed',elapsed)
                    else:
                        age=self.age_fun(face_img)
                        elapsed = time.time() - t
                        if self.print_stat:
                            print('age elapsed',elapsed)
                        t = time.time()
                        gender=self.gender_fun(face_img)
                        elapsed = time.time() - t
                        if self.print_stat:
                            print('gender elapsed',elapsed)
                        features=[]
                    avg_age+=age
                    avg_gender+=gender
                age=avg_age/len(boxes)
                gender=avg_gender/len(boxes)
                ages.append(age)
                genders.append(gender)
                facial_features.append(features)
                bboxes.append(boxes[0])
        return bboxes, points,ages,genders,facial_features
    
    def show_detection_results(self,draw):
        bounding_boxes, points,ages,genders,facial_features=self.process_image(draw)
        
        for i,b in enumerate(bounding_boxes):
            b=[int(bi) for bi in b]
            x1,y1,x2,y2=b[0:4]
            
            age='%.0f'%(ages[i])
            is_male=FacialImageProcessing.is_male(genders[i])
            if is_male:
                clr=(255, 0, 0)
            else: #female
                clr=(0, 0, 255)
            cv2.rectangle(draw, (x1, y1), (x2, y2), clr)
            cv2.putText(draw,age, (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
        
        #nose,right eye,left eye,right mouth,left mouth
        #colors=[(0, 0, 255),(0, 255, 0),(255, 0, 0),(0, 255, 255),(255, 255, 0)]
        #for p in points.T:
        #    for i in range(5):
        #        cv2.circle(draw, (p[i], p[i + 5]), 1, colors[i], 2)
        return draw,len(bounding_boxes)
    
    @staticmethod
    def load_graph_def(frozen_graph_filename):
        graph_def=None
        with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        return graph_def
    
    @staticmethod
    def load_graph(frozen_graph_filename, prefix=''):
        graph_def = FacialImageProcessing.load_graph_def(frozen_graph_filename)
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name=prefix)
        return graph

    @staticmethod
    def load_mtcnn(sess,graph):
        pnet_out_1=graph.get_tensor_by_name('pnet/conv4-2/BiasAdd:0')
        pnet_out_2=graph.get_tensor_by_name('pnet/prob1:0')
        pnet_in=graph.get_tensor_by_name('pnet/input:0')
        
        rnet_out_1=graph.get_tensor_by_name('rnet/conv5-2/conv5-2:0')
        rnet_out_2=graph.get_tensor_by_name('rnet/prob1:0')
        rnet_in=graph.get_tensor_by_name('rnet/input:0')
        
        onet_out_1=graph.get_tensor_by_name('onet/conv6-2/conv6-2:0')
        onet_out_2=graph.get_tensor_by_name('onet/conv6-3/conv6-3:0')
        onet_out_3=graph.get_tensor_by_name('onet/prob1:0')
        onet_in=graph.get_tensor_by_name('onet/input:0')
        
        pnet_fun = lambda img : sess.run((pnet_out_1, pnet_out_2), feed_dict={pnet_in:img})
        rnet_fun = lambda img : sess.run((rnet_out_1, rnet_out_2), feed_dict={rnet_in:img})
        onet_fun = lambda img : sess.run((onet_out_1, onet_out_2, onet_out_3), feed_dict={onet_in:img})
        return pnet_fun, rnet_fun, onet_fun
        
    @staticmethod
    def bbreg(boundingbox,reg):
        # calibrate bounding boxes
        if reg.shape[1]==1:
            reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

        w = boundingbox[:,2]-boundingbox[:,0]+1
        h = boundingbox[:,3]-boundingbox[:,1]+1
        b1 = boundingbox[:,0]+reg[:,0]*w
        b2 = boundingbox[:,1]+reg[:,1]*h
        b3 = boundingbox[:,2]+reg[:,2]*w
        b4 = boundingbox[:,3]+reg[:,3]*h
        boundingbox[:,0:4] = np.transpose(np.vstack([b1, b2, b3, b4 ]))
        return boundingbox
     
    @staticmethod
    def generateBoundingBox(imap, reg, scale, t):
        # use heatmap to generate bounding boxes
        stride=2
        cellsize=12

        imap = np.transpose(imap)
        dx1 = np.transpose(reg[:,:,0])
        dy1 = np.transpose(reg[:,:,1])
        dx2 = np.transpose(reg[:,:,2])
        dy2 = np.transpose(reg[:,:,3])
        y, x = np.where(imap >= t)
        if y.shape[0]==1:
            dx1 = np.flipud(dx1)
            dy1 = np.flipud(dy1)
            dx2 = np.flipud(dx2)
            dy2 = np.flipud(dy2)
        score = imap[(y,x)]
        reg = np.transpose(np.vstack([ dx1[(y,x)], dy1[(y,x)], dx2[(y,x)], dy2[(y,x)] ]))
        if reg.size==0:
            reg = np.empty((0,3))
        bb = np.transpose(np.vstack([y,x]))
        q1 = np.fix((stride*bb+1)/scale)
        q2 = np.fix((stride*bb+cellsize-1+1)/scale)
        boundingbox = np.hstack([q1, q2, np.expand_dims(score,1), reg])
        return boundingbox, reg
     
    # function pick = nms(boxes,threshold,type)
    @staticmethod
    def nms(boxes, threshold, method):
        if boxes.size==0:
            return np.empty((0,3))
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        s = boxes[:,4]
        area = (x2-x1+1) * (y2-y1+1)
        I = np.argsort(s)
        pick = np.zeros_like(s, dtype=np.int16)
        counter = 0
        while I.size>0:
            i = I[-1]
            pick[counter] = i
            counter += 1
            idx = I[0:-1]
            xx1 = np.maximum(x1[i], x1[idx])
            yy1 = np.maximum(y1[i], y1[idx])
            xx2 = np.minimum(x2[i], x2[idx])
            yy2 = np.minimum(y2[i], y2[idx])
            w = np.maximum(0.0, xx2-xx1+1)
            h = np.maximum(0.0, yy2-yy1+1)
            inter = w * h
            if method is 'Min':
                o = inter / np.minimum(area[i], area[idx])
            else:
                o = inter / (area[i] + area[idx] - inter)
            I = I[np.where(o<=threshold)]
        pick = pick[0:counter]
        return pick

    # function [dy edy dx edx y ey x ex tmpw tmph] = pad(total_boxes,w,h)
    @staticmethod
    def pad(total_boxes, w, h):
        # compute the padding coordinates (pad the bounding boxes to square)
        tmpw = (total_boxes[:,2]-total_boxes[:,0]+1).astype(np.int32)
        tmph = (total_boxes[:,3]-total_boxes[:,1]+1).astype(np.int32)
        numbox = total_boxes.shape[0]

        dx = np.ones((numbox), dtype=np.int32)
        dy = np.ones((numbox), dtype=np.int32)
        edx = tmpw.copy().astype(np.int32)
        edy = tmph.copy().astype(np.int32)

        x = total_boxes[:,0].copy().astype(np.int32)
        y = total_boxes[:,1].copy().astype(np.int32)
        ex = total_boxes[:,2].copy().astype(np.int32)
        ey = total_boxes[:,3].copy().astype(np.int32)

        tmp = np.where(ex>w)
        edx.flat[tmp] = np.expand_dims(-ex[tmp]+w+tmpw[tmp],1)
        ex[tmp] = w
        
        tmp = np.where(ey>h)
        edy.flat[tmp] = np.expand_dims(-ey[tmp]+h+tmph[tmp],1)
        ey[tmp] = h

        tmp = np.where(x<1)
        dx.flat[tmp] = np.expand_dims(2-x[tmp],1)
        x[tmp] = 1

        tmp = np.where(y<1)
        dy.flat[tmp] = np.expand_dims(2-y[tmp],1)
        y[tmp] = 1
        
        return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph

    # function [bboxA] = rerec(bboxA)
    @staticmethod
    def rerec(bboxA):
        # convert bboxA to square
        h = bboxA[:,3]-bboxA[:,1]
        w = bboxA[:,2]-bboxA[:,0]
        l = np.maximum(w, h)
        bboxA[:,0] = bboxA[:,0]+w*0.5-l*0.5
        bboxA[:,1] = bboxA[:,1]+h*0.5-l*0.5
        bboxA[:,2:4] = bboxA[:,0:2] + np.transpose(np.tile(l,(2,1)))
        return bboxA

    def mtcnn_detect_faces(self,img):
        # im: input image
        # threshold: threshold=[th1 th2 th3], th1-3 are three steps's threshold
        threshold = [ 0.6, 0.7, 0.9 ]  # three steps's threshold
        # fastresize: resize img from last scale (using in high-resolution images) if fastresize==true
        factor = 0.709 # scale factor
        factor_count=0
        total_boxes=np.empty((0,9))
        points=np.array([])
        h=img.shape[0]
        w=img.shape[1]
        minl=np.amin([h, w])
        m=12.0/self.minsize
        minl=minl*m
        # creat scale pyramid
        scales=[]
        while minl>=12:
            scales += [m*np.power(factor, factor_count)]
            minl = minl*factor
            factor_count += 1

        # first stage
        #t=time.time()
        for j in range(len(scales)):
            scale=scales[j]
            hs=int(np.ceil(h*scale))
            ws=int(np.ceil(w*scale))
            im_data = cv2.resize(img, (ws,hs), interpolation=cv2.INTER_AREA)
            im_data = (im_data-127.5)*0.0078125
            img_x = np.expand_dims(im_data, 0)
            img_y = np.transpose(img_x, (0,2,1,3))
            out = self.pnet(img_y)
            out0 = np.transpose(out[0], (0,2,1,3))
            out1 = np.transpose(out[1], (0,2,1,3))
            
            boxes, _ = FacialImageProcessing.generateBoundingBox(out1[0,:,:,1].copy(), out0[0,:,:,:].copy(), scale, threshold[0])
            
            # inter-scale nms
            pick = FacialImageProcessing.nms(boxes.copy(), 0.5, 'Union')
            if boxes.size>0 and pick.size>0:
                boxes = boxes[pick,:]
                total_boxes = np.append(total_boxes, boxes, axis=0)
        numbox = total_boxes.shape[0]
        #elapsed = time.time() - t
        #print('1 phase nb=%d elapsed=%f'%(numbox,elapsed))
        if numbox>0:
            pick = FacialImageProcessing.nms(total_boxes.copy(), 0.7, 'Union')
            total_boxes = total_boxes[pick,:]
            regw = total_boxes[:,2]-total_boxes[:,0]
            regh = total_boxes[:,3]-total_boxes[:,1]
            qq1 = total_boxes[:,0]+total_boxes[:,5]*regw
            qq2 = total_boxes[:,1]+total_boxes[:,6]*regh
            qq3 = total_boxes[:,2]+total_boxes[:,7]*regw
            qq4 = total_boxes[:,3]+total_boxes[:,8]*regh
            total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:,4]]))
            total_boxes = FacialImageProcessing.rerec(total_boxes.copy())
            total_boxes[:,0:4] = np.fix(total_boxes[:,0:4]).astype(np.int32)
            dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = FacialImageProcessing.pad(total_boxes.copy(), w, h)

        numbox = total_boxes.shape[0]
        #elapsed = time.time() - t
        #print('2 phase nb=%d elapsed=%f'%(numbox,elapsed))
        if numbox>0:
            # second stage
            tempimg = np.zeros((24,24,3,numbox))
            for k in range(0,numbox):
                tmp = np.zeros((int(tmph[k]),int(tmpw[k]),3))
                tmp[dy[k]-1:edy[k],dx[k]-1:edx[k],:] = img[y[k]-1:ey[k],x[k]-1:ex[k],:]
                if tmp.shape[0]>0 and tmp.shape[1]>0 or tmp.shape[0]==0 and tmp.shape[1]==0:
                    tempimg[:,:,:,k] = cv2.resize(tmp, (24,24), interpolation=cv2.INTER_AREA)
                else:
                    return np.empty()
            tempimg = (tempimg-127.5)*0.0078125
            tempimg1 = np.transpose(tempimg, (3,1,0,2))
            out = self.rnet(tempimg1)
            out0 = np.transpose(out[0])
            out1 = np.transpose(out[1])
            score = out1[1,:]
            ipass = np.where(score>threshold[1])
            total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(), np.expand_dims(score[ipass].copy(),1)])
            mv = out0[:,ipass[0]]
            if total_boxes.shape[0]>0:
                pick = FacialImageProcessing.nms(total_boxes, 0.7, 'Union')
                total_boxes = total_boxes[pick,:]
                total_boxes = FacialImageProcessing.bbreg(total_boxes.copy(), np.transpose(mv[:,pick]))
                total_boxes = FacialImageProcessing.rerec(total_boxes.copy())

        numbox = total_boxes.shape[0]
        #elapsed = time.time() - t
        #print('3 phase nb=%d elapsed=%f'%(numbox,elapsed))
        if numbox>0:
            # third stage
            total_boxes = np.fix(total_boxes).astype(np.int32)
            dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = FacialImageProcessing.pad(total_boxes.copy(), w, h)
            tempimg = np.zeros((48,48,3,numbox))
            for k in range(0,numbox):
                tmp = np.zeros((int(tmph[k]),int(tmpw[k]),3))
                tmp[dy[k]-1:edy[k],dx[k]-1:edx[k],:] = img[y[k]-1:ey[k],x[k]-1:ex[k],:]
                if tmp.shape[0]>0 and tmp.shape[1]>0 or tmp.shape[0]==0 and tmp.shape[1]==0:
                    tempimg[:,:,:,k] = cv2.resize(tmp, (48,48), interpolation=cv2.INTER_AREA)
                else:
                    return np.empty()
            tempimg = (tempimg-127.5)*0.0078125
            tempimg1 = np.transpose(tempimg, (3,1,0,2))
            out = self.onet(tempimg1)
            out0 = np.transpose(out[0])
            out1 = np.transpose(out[1])
            out2 = np.transpose(out[2])
            score = out2[1,:]
            points = out1
            ipass = np.where(score>threshold[2])
            points = points[:,ipass[0]]
            total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(), np.expand_dims(score[ipass].copy(),1)])
            mv = out0[:,ipass[0]]

            w = total_boxes[:,2]-total_boxes[:,0]+1
            h = total_boxes[:,3]-total_boxes[:,1]+1
            points[0:5,:] = np.tile(w,(5, 1))*points[0:5,:] + np.tile(total_boxes[:,0],(5, 1))-1
            points[5:10,:] = np.tile(h,(5, 1))*points[5:10,:] + np.tile(total_boxes[:,1],(5, 1))-1
            if total_boxes.shape[0]>0:
                total_boxes = FacialImageProcessing.bbreg(total_boxes.copy(), np.transpose(mv))
                pick = FacialImageProcessing.nms(total_boxes.copy(), 0.7, 'Min')
                total_boxes = total_boxes[pick,:]
                points = points[:,pick]
        #elapsed = time.time() - t
        #print('4 phase elapsed=%f'%(elapsed))            
        return total_boxes, points
    

def show_webcam():
    cam = cv2.VideoCapture(0)
    imgProcessing=FacialImageProcessing(True)
    while True:
        _, draw = cam.read()
        draw,_=imgProcessing.show_detection_results(draw)
        cv2.imshow('my webcam', draw)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()
    imgProcessing.close()

def get_video_file_orientation(video_filepath):
    rotation=0
    try:
        cmd = os.path.join(os.path.dirname(os.path.abspath(__file__)),'exiftool')+' -t -Rotation %s' % video_filepath
        p = subprocess.Popen(
            cmd.split(' '),
            stdout=subprocess.PIPE
        )
        str=p.stdout.read().decode('utf-8').strip()

        reo_rotation = re.compile('Rotation\s(?P<rotation>.*)', re.IGNORECASE)
        match_rotation = reo_rotation.search(str)
        if match_rotation is not None:
            rotation = int(match_rotation.groups()[0])
    except:
        print ('Unexpected error:', sys.exc_info()[0],cmd)
    return rotation

def show_video(video_filepath):
    if True: #detect rotation
        rotation=get_video_file_orientation(video_filepath)
        print('rotation:',rotation)
    imgProcessing=FacialImageProcessing(True)
    counter=0
    delta_counter=5
    video = cv2.VideoCapture(video_filepath)
    while video.isOpened():
        #_, draw = video.read()
        if video.grab()==0:
            break
        counter+=1
        if counter%delta_counter!=0:
            continue
        _,draw=video.retrieve()
        height, width, channels = draw.shape
        if width>640 or height>480:
            draw=cv2.resize(draw, (min(width,640),min(height,480)))
        if rotation==90:
            draw=cv2.transpose(draw)
            draw=cv2.flip(draw,1)
        elif rotation==270:
            draw=cv2.transpose(draw)
            draw=cv2.flip(draw,0)
        
        draw,num_faces=imgProcessing.show_detection_results(draw)
        #delta_counter=3 if num_faces==0 else 1
        cv2.imshow('my webcam', draw)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()
    imgProcessing.close()
    
def process_all_images(args, save_processed=False):
    imgProcessing=FacialImageProcessing(True)
    #sess = tf.Session()#config=tf.ConfigProto(device_count={'CPU':1,'GPU':0}))
    
    for filename in args:
        draw = cv2.imread(filename)
        height, width, channels = draw.shape
        if width>640 or height>480:
            draw=cv2.resize(draw, (min(width,640),min(height,480)))
        #draw=cv2.resize(draw, (192,192))
        draw,_=imgProcessing.show_detection_results(draw)
        cv2.imshow(filename, draw)

        if save_processed:
            fn,ext=os.path.splitext(filename)
            output_filename = fn+'_det.jpg'
            cv2.imwrite(output_filename,draw)
        cv2.waitKey(-1)
        cv2.destroyAllWindows()
    
    imgProcessing.close()

    
if __name__ == '__main__':
    #main(parse_arguments(sys.argv[1:]))
    if len(sys.argv)==2:
        show_video(sys.argv[1])
    elif len(sys.argv)>2:
        process_all_images(sys.argv[1:])
    else:
        show_webcam()
