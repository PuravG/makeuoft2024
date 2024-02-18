#intial imports: 
import cv2
import mediapipe as mp

import numpy as np
from numpy import asarray
# import pandas as pd
# import tensorflow as tf
import os
import csv
# from tensorflow.keras.losses import SparseCategoricalCrossentropy
# from tensorflow.keras.optimizers import Adam
import time
# import streamlit as st
from matplotlib import pyplot as plt
from PIL import Image

import json
import requests
 


def make_prediction(model_path, labels, csv_file):
    my_model = tf.keras.models.load_model(model_path, compile=False)
    my_model.compile(loss=SparseCategoricalCrossentropy(), optimizer=Adam(), metrics=["accuracy"])
    predictions = {n: 0 for n in labels}
    csv = pd.read_csv(csv_file)
    # print(csv)
    # nevermind, I want to predict for each frame.
    coords = np.array(csv)

    # so yea, this is going to iterate once.
    for frame in coords:
        new_frame = tf.expand_dims(frame,0)
        preds = my_model.predict(new_frame)
        pred_value = np.argmax(preds)
        # for i in range(len(preds[0])):
        #     print(preds[0][i])
        #     predictions[saved_classes[i]] += preds[0][i]
        #     total += preds[0][i]
        predictions[labels[pred_value]] += 1
        # print(preds)
    final_prediction = max(predictions, key=predictions.get)
    return predictions, final_prediction

MODEL_PATH = r"D:\Personnel\Other learning\Programming\Personal_projects\3_Hackathons_with_buddies\MAKEUOFT2024\drowsiness_detection\best_models\eye\model.114-0.94"
def predict_eye(file_name):
    predictions, final_prediction = make_prediction(MODEL_PATH, ["closed", "open"], file_name)
    # final_prediction is either 0 or 1
    return final_prediction

MODEL_PATH = r"D:\Personnel\Other learning\Programming\Personal_projects\3_Hackathons_with_buddies\MAKEUOFT2024\drowsiness_detection\best_models\yawn\model.07-0.95"
def predict_yawn(file_name):
    predictions, final_prediction = make_prediction(MODEL_PATH, ["no_yawn", "yawn"], file_name)
    return final_prediction



if __name__ == '__main__':
    final_prediction_eye = predict_eye("eye.csv")
    final_prediction_yawn = predict_yawn("yawn.csv")


    # writing the values for prediction into a json file
    with open("predictions.json", mode='a') as f:
        json.dump([final_prediction_eye, final_prediction_yawn], f)
