'''
This is the implementation of Mask R-CNN based on NVIDIA GitHub
The reference link is: https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN

There is another implementation of Mask R-CNN by Matterport. However the TensorFlow.1x codes are outdated.

The main reason to use NVIDIA implementation is to further deploy in Jetson Nano. 
In addition, TensorRT (by NVIDIA) has been proven to run significantly faster than Google's TensorLite.
'''
import os