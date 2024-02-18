import json
import requests
import cv2
import numpy as np
from numpy import asarray
import pandas as pd
import tensorflow as tf
import mediapipe as mp
import os
import csv
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import time
from PIL import Image
 

def make_prediction(model_path, labels, csv_file):
    my_model = tf.keras.models.load_model(model_path, compile=True)
    my_model.compile(loss=SparseCategoricalCrossentropy(), optimizer=Adam(), metrics=["accuracy"])
    predictions = {n: 0 for n in labels}

    # ! THIS WILL SKIP THE FIRST ROW UNLESS I WRITER header = None, which explains why I had an empty array
    csv = pd.read_csv(csv_file, header = None)
    # print(csv)
    # # nevermind, I want to predict for each frame.
    coords = np.array(csv)
    # print("Hello", coords.shape)
    # coords = coords
    # # print(coords.shape)
    # for frame in coords:
    #     print("HI")
    #     print(frame)

    # coords.reshape((None, 36))

    # so yea, this is going to iterate once.
    # for frame in coords:
    #     new_frame = tf.expand_dims(frame,0)
    #     preds = my_model.predict(new_frame)
    #     pred_value = np.argmax(preds)
    #     for i in range(len(preds[0])):
    #         print(preds[0][i])
    #         predictions[saved_classes[i]] += preds[0][i]
    #         total += preds[0][i]
    #     predictions[labels[pred_value]] += 1
    #     print(preds)
    
    # since I am not looping through multiple frame (i only have 1) then I will simply take the first value
    new_frame = tf.expand_dims(coords[0],0)
    preds = my_model.predict(new_frame)
    pred_value = np.argmax(preds)
    # final_prediction = max(predictions, key=predictions.get)
    return labels[pred_value]

def predict_eye(file_name):
    MODEL_PATH = r"D:\Personnel\Other learning\Programming\Personal_projects\3_Hackathons_with_buddies\MAKEUOFT2024\drowsiness_detection\best_models\eye\model.133-0.99"
    # predictions, final_prediction = 
    final_prediction = make_prediction(MODEL_PATH, ["closed", "open"], file_name)
    # final_prediction is either 0 or 1
    return final_prediction

def predict_yawn(file_name):
    MODEL_PATH = r"D:\Personnel\Other learning\Programming\Personal_projects\3_Hackathons_with_buddies\MAKEUOFT2024\drowsiness_detection\best_models\yawn\model.139-0.90"
    final_prediction = make_prediction(MODEL_PATH, ["no_yawn", "yawn"], file_name)
    return final_prediction