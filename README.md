# HSE_FaceRec_tf
Tensorflow/Keras small models for face recognition.

- MobileNet-192 (vgg2_mobilenet_2.h5 in Keras and identical vgg2_mobilenet_2.pb in Tensorflow). Model size 13 MB.
- ResNet-50 (vgg2_resnet.pb). Model size 95 MB.

These models were trained on a training set from [VGGFace2 dataset](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)

We tested these models as feature extractors in 1-NN (nearest neighbour) method with 50% train/test split of several known facial datasets including:
- 9164 photos of 1680 persons from LFW (Labelled Faces in the Wild) with more than one photo. Face *identification* accuracy: 94.8% (MobileNet), 98.6% (ResNet)
- 5396 still photos of 500 subjects from img folder of IJB-A dataset. Face *identification* accuracy: 88.7% (MobileNet), 90.1% (ResNet)
