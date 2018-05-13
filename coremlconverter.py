#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 13:35:16 2018

@author: KaranJaisingh
"""

# import necessary packages
from keras.models import load_model
import coremltools
import argparse
import pickle
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
args = vars(ap.parse_args())
# arguments passed:
    #1. --model: path to model
    #2. --labelbin: path to class label binarizer
    
# load the class labels
print("[INFO] loading class labels from label binarizer")
lb = pickle.loads(open(args["labelbin"], "rb").read())
class_labels = lb.classes_.tolist()
print("[INFO] class labels: {}".format(class_labels))
 
# load the trained convolutional neural network
print("[INFO] loading model...")
model = load_model(args["model"])

# convert the model to coreml format
print("[INFO] converting model")
coreml_model = coremltools.converters.keras.convert(model,
	input_names="image",
	image_input_names="image",
	image_scale=1/255.0, # very important parameter
	class_labels=class_labels, # obtained from LabelBinarizer object
	is_bgr=True) # extremely important - must be set to true is images trained with BGR colours

# save the model to disk
output = args["model"].rsplit(".", 1)[0] + ".mlmodel" # change the extension of model
print("[INFO] saving model as {}".format(output))
coreml_model.save(output)

# To run the script in Terminal:
# python coremlconverter.py --model <MODEL_NAME>.model --labelbin lb.pickle