# HSE_FaceRec_tf
Tensorflow/Keras small models for face recognition.

- MobileNet-192 (vgg2_mobilenet_2.h5 in Keras and identical vgg2_mobilenet_2.pb in Tensorflow). Model size 13 MB.
- ResNet-50 (vgg2_resnet.pb). Model size 95 MB.

These models were trained on a training set from [VGGFace2 dataset](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) using Softmax loss

Please put the corrct path to the LFW directory in DATASET_PATH (line 26 of facerec_text.py).
Out simple testing script supports [FaceNet, Inception ResNet v1](https://github.com/davidsandberg/facenet) and [InsightFace (ArcFace)](https://github.com/AIInAi/tf-insightface) models. Please download them from corresponding repositories

We tested these models as feature extractors in 1-NN (nearest neighbour) method with 50% train/test split of several known facial datasets including:
- (UPDATED) 3739 photos of 596 persons from the intersection of LFW (Labeled Faces in the Wild) and YTF datasets with more than one photo. Face *identification* accuracy (single training image per class): 92.1% (MobileNet), 97.8% (VGG2 ResNet), 96.6% (FaceNet), 88.9% (InsightFace)
- 9164 photos of 1680 persons from LFW with more than one photo. Face *identification* accuracy (train/test split 0.5): 94.8% (MobileNet), 98.8% (VGG2 ResNet), 97.7% (FaceNet), 92.6% (InsightFace)
- 5396 still photos of 500 subjects from img folder of IJB-A dataset. Face *identification* accuracy: 88.7% (MobileNet), 90.1% (VGG2 ResNet)
