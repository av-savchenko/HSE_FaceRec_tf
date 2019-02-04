#Efficient facial representations for age, gender and identity recognition in organizing photo albums using multi-output ConvNet

Source code for facial analysis including gender recognition, age predicition, clustering of similar faces and extraction of potential public photos

We provide three pre-trained models for simulatenous age/gender recognition:
- age_gender_tf2_new-01-0.14-0.92: here we freeze the weights of the MobileNet base pre-trained on VGGFace2 for face identification
- age_gender_tf2_new-01-0.14-0.92_quantized: the first model quantized by the Tensorflow graph_transform
- age_gender_tf2_new-01-0.16-0.92: the first model, all weights of which were fine-tuned for age/gender recognition, i.e., the identity features became less informative
The model can be changed in the facial_analysis.py (line 40)

Change InputDirectory property in config.txt to point to the correct direcotiry with photos (jpg/png) and video clips (mov/avi)

Required libraries: Tensorflow, OpenCV, SciPy, SKLearn, MatplotLib

Please download [exiftool](https://www.sno.phy.queensu.ca/~phil/exiftool/) in order to detect orientation of video clips