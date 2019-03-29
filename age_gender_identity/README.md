# Efficient facial representations for age, gender and identity recognition in organizing photo albums using multi-output ConvNet

Source code for facial analysis including gender recognition, age predication, clustering of similar faces and extraction of potential public photos

We provide three pre-trained models for simultaneous age/gender recognition:
1. age_gender_tf2_new-01-0.14-0.92: here we freeze the weights of the MobileNet base pre-trained on VGGFace2 for face identification
2. age_gender_tf2_new-01-0.14-0.92_quantized: the first model quantized by the Tensorflow graph_transform
3. age_gender_tf2_new-01-0.16-0.92: the first model, all weights of which were fine-tuned for age/gender recognition, i.e., the identity features became less informative
4. age_gender_tf2_224_deep-03-0.13-0.97: similar to the first model but with slightly higher accuracy
The model can be changed in the facial_analysis.py (line 45)

Change InputDirectory property in config.txt to point to the correct directory with photos (jpg/png) and video clips (mov/avi)

Required libraries: Tensorflow, OpenCV, SciPy, SKLearn, MatplotLib, MXNet (for Insightface)

Please download [exiftool](https://www.sno.phy.queensu.ca/~phil/exiftool/) in order to detect orientation of video clips

Script utkface_test.py contains comparison of our models with several known open-source implementations of age/gender recognition for [UTKFace dataset](https://susanqq.github.io/UTKFace/)

* [DEX VGG](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) models converted from Caffe to Tensorflow
* [Wide ResNet (weights.28-3.73.hdf5)](https://github.com/yu4u/age-gender-estimation)
* [Wide ResNet (weights.18-4.06.hdf5)](https://github.com/Tony607/Keras_age_gender)
* [FaceNet](https://github.com/BoyuanJiang/Age-Gender-Estimate-TF)
* [BKNetStyle2](https://github.com/truongnmt/multi-task-learning)
* [SSRNet](https://github.com/shamangary/SSR-Net)
* [MobileNet v2 (Agegendernet)](https://github.com/dandynaufaldi/Agendernet)
* Two models from [Insightface](https://github.com/deepinsight/insightface/)
* [Inception trained on Adience](https://github.com/dpressel/rude-carnie)

Our best models (age_gender_tf2_224_deep-03-0.13-0.97 and age_gender_tf2_224_deep_fn-01-0.15-0.98) significantly outperforms all the above-mentioned models. We achieve 91.9% gender recognition accuracy and 6.00/5.96 age prediction MAE for [aligned and cropped faces](https://drive.google.com/drive/folders/0BxYys69jI14kU0I1YUQyY1ZDRUE) from UTKFace.

In addition, we used the testing set from the SOTA paper ["Consistent Rank Logits for Ordinal Regression with Convolutional Neural Networks"](https://arxiv.org/pdf/1901.07884.pdf) which uses subset of UTKFace dataset with age ranges [21,60]. Our models achieve 97.5% gender recognition accuracy and 5.39/5.36 age prediction MAE. It is lower than 5.47 MAE of the best CORAL-CNN model from this paper, which was additionally trained on another subset of UTKFace. Our models do not require such fine-tuning.

Finally, we performed pre-processing of initial [In-the-wild faces](https://drive.google.com/drive/folders/0BxYys69jI14kSVdWWllDMWhnN2c) from UTKFace using pipeline from [Agegendernet](https://github.com/dandynaufaldi/Agendernet), which includes face detection and alignment using dlib with margin 0.4. All 648 images without exactly one detected faces are removed from the testing set. Such setup significantly improves the results of existing methods. Anyway, our models remain much better. They achieve 93.8%/94.1% gender recognition accuracy and 5.74/5.44 age prediction MAE for complete UTKFace, and 97.9%/98.2% gender accuracy with 5.17/4.90 age prediction MAE for the testing set from the above-mentioned [paper](https://arxiv.org/pdf/1901.07884.pdf).