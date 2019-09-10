from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import mxnet as mx
import random
import cv2
import sklearn
sys.path.append('D:/src_code/DNN_models/age_gender/insightface/src/common')
import face_preprocess


def do_flip(data):
  for idx in range(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

class InsightFaceModel:
  def __init__(self):
    image_size = 112
    self.image_size='{},{}'.format(image_size,image_size)
    self.flip=0
    model_path='D:/src_code/DNN_models/age_gender/insightface/models/model-r100-arcface-ms1m-refine-v2/model-r100-ii/model,0'
    _vec = model_path.split(',')
    prefix = _vec[0]
    epoch = int(_vec[1])
    print('loading',prefix, epoch)
    ctx = mx.gpu(0)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers['fc1_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
    model.bind(data_shapes=[('data', (1, 3, image_size, image_size))])
    model.set_params(arg_params, aux_params)
    self.model = model


  def get_feature(self, face_img):
    nimg = face_preprocess.preprocess(face_img, None, None, image_size=self.image_size)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2,0,1))
    #print(nimg.shape)
    embedding = None
    for flipid in [0,1]:
      if flipid==1:
        if self.flip==0:
          break
        do_flip(aligned)
      input_blob = np.expand_dims(aligned, axis=0)
      data = mx.nd.array(input_blob)
      db = mx.io.DataBatch(data=(data,))
      self.model.forward(db, is_train=False)
      _embedding = self.model.get_outputs()[0].asnumpy()
      #print(_embedding.shape)
      if embedding is None:
        embedding = _embedding
      else:
        embedding += _embedding
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding

