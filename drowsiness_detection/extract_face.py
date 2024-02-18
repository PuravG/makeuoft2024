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

from predict import *

#!! make sure you are running the python files from the current folder where eye.csv are

abs_path = os.getcwd()
 
def setup_file(csv_file, num_dimension = 3, num_coords = 33 + 21 + 21):
    rows = []
    #creating empty file in folder, i added the start_time in the name of the csv file, so that if a symbol appears many times in a video, it will still be created in two different csv files, just that they will have different starting times
    # csv_file = f"/users/aly/documents/programming/apps/machine learning/asl converter/training_models/mediapipe/demo_test/demo.csv"
    # csv_file="d:/personnel/other learning/programming/personal_projects/asl_language_translation/training_models/mediapipe/demo_test/demo.csv"
    # if os.path.exists(csv_file):
    #     return 



# Setup CSV File for the videos
# 21 right hand landmarks, 21 left hand landmarks, 33 pose landmarks
    # num_coords = 33 + 21 + 21
    landmarks = []

    # we are only working with x, y
    if num_dimension == 2:
        for val in range(1, num_coords+1):
            landmarks += ['x{}'.format(val), 'y{}'.format(val)]#.format(val), 'z{}'.format(val)]#, 'v{}'.format(val)]
    
    elif num_dimension == 3:
        for val in range(1, num_coords+1):
            landmarks += ['x{}'.format(val), 'y{}'.format(val).format(val), 'z{}'.format(val)]#, 'v{}'.format(val)]
    
    # I will assume they just want all the coordinates:
    else:
        landmarks += ['x{}'.format(val), 'y{}'.format(val).format(val), 'z{}'.format(val), 'v{}'.format(val)]

    print("Initialized an empty landmarks of size:", len(landmarks))

    with open(csv_file, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)

def capture_nose(image, face_landmarks, csv_file):
    # Get nose tip coordinates.
    nose_tip = face_landmarks.landmark[1]
    x, y = int(nose_tip.x * image.shape[1]), int(nose_tip.y * image.shape[0])
    
    # Print nose tip coordinates.
    # print(f"Nose Tip Coordinates: (X: {x}, Y: {y})")


    # writing into the correct csv file
    row = [x, y]
    index = 0

    with open(csv_file, mode='a', newline='') as f:
        # basically, Krish wants the last value only, so i'll delete everything in the file
        f.truncate(0)
        json.dump(row, f)
    #     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #                     #   for row in rows:
    #     # writerow expects a list
    #     csv_writer.writerow(row) 
    return x, y

mp_hands = mp.solutions.hands
mp_facemesh = mp.solutions.face_mesh
mp_drawing  = mp.solutions.drawing_utils
denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates
 
# Landmark points corresponding to left eye
all_left_eye_idxs = list(mp_facemesh.FACEMESH_LEFT_EYE)
# flatten and remove duplicates
all_left_eye_idxs = set(np.ravel(all_left_eye_idxs)) 
 
# Landmark points corresponding to right eye
all_right_eye_idxs = list(mp_facemesh.FACEMESH_RIGHT_EYE)
all_right_eye_idxs = set(np.ravel(all_right_eye_idxs))
 
# Combined for plotting - Landmark points for both eye
all_idxs = all_left_eye_idxs.union(all_right_eye_idxs)
 
# The chosen 12 points:   P1,  P2,  P3,  P4,  P5,  P6
chosen_left_eye_idxs  = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33,  160, 158, 133, 153, 144]
all_chosen_idxs = chosen_left_eye_idxs + chosen_right_eye_idxs

def capture_eye(image, face_landmarks, file_name):
    imgH, imgW, _ = image.shape

    # this is an example code where the person gets the nose
    # landmark_0 = results.multi_face_landmarks[0].landmark[0]

    # note: face_landmarks = results.multi_face_landmarks[i], just at different indexes
    rows = []
    index = 0
    for index in range(len(face_landmarks.landmark)):
        if index in chosen_left_eye_idxs or index in chosen_right_eye_idxs:
            # if index == 263:
            #     print("YO")
            #     print("Total Length of '.landmark':", len(face_landmarks.landmark))
            
            current_point = face_landmarks.landmark[index]
            # print(current_point)
            landmark_x = current_point.x * imgW 
            landmark_y = current_point.y * imgH
            landmark_z = current_point.z * imgW #
                
            # print()
            # print("X:", landmark_x)
            # print("Y:", landmark_y)
            # print("Z:", landmark_z)
            

            row = [landmark_x, landmark_y, landmark_z]
            # so yea, I am adding elements to the list called rows
            rows += row  

    
        

    # writing into the correct csv file
    with open(file_name, mode='a', newline='') as f:
        # I want to erase everything before inferring -- if we are in prediction mode
        f.truncate(0)

        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        #   for row in rows:
        # writerow expects a list
        csv_writer.writerow(rows) 
        # print("The length of the row is ", len(rows))

# set of mediapipe points that represent the facial features for a yawn
faceConnect = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 
400, 378, 379, 365, 397, 367, 
288, 435, 361, 401, 323, 366, 451, #face contours
57, 77, 89, 88, 178, 87, 14, 317, 402, 319, 307, #mouth
64, 59, 44, 1, 457, 294,  #straight nose
1, 4, 5, 195, 197, 6, 168, 8,  #vertical up
190, 222, 224, 124, #left eye
413, 441, 442, 443, 445 #right eye
]

def capture_yawn(image, face_landmarks, file_name):
    imgH, imgW, _ = image.shape

    # this is an example code where the person gets the nose
    # landmark_0 = results.multi_face_landmarks[0].landmark[0]

    # note: face_landmarks = results.multi_face_landmarks[i], just at different indexes
    rows = []
    index = 0
    for index in range(len(face_landmarks.landmark)):
        if index in faceConnect:
            current_point = face_landmarks.landmark[index]
            # print(current_point)
            landmark_x = current_point.x * imgW 
            landmark_y = current_point.y * imgH
            landmark_z = current_point.z * imgW #
                
            # print()
            # print("X:", landmark_x)
            # print("Y:", landmark_y)
            # print("Z:", landmark_z)
            

            row = [landmark_x, landmark_y, landmark_z]
            # so yea, I am adding elements to the list called rows
            rows += row  

    
        

    # writing into the correct csv file
    with open(file_name, mode='a', newline='') as f:
        # I want to erase everything before inferring -- if we are in prediction mode
        f.truncate(0)

        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        #   for row in rows:
        # writerow expects a list
        csv_writer.writerow(rows) 
        # print("The length of the row is ", len(rows))

def open_cam():   
  # webcam input 
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    cap = cv2.VideoCapture(0)  # 0 for the default webcam

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        # to track the frame number, and skip some frames.
        frame_number = 0

        #this needs to be outside the while loop!
        previous = 0
        curr = 0
        up_or_down = 0
        while cap.isOpened():

            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            
            # Convert the BGR image to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # To improve performance, mark the image as not writeable.
            image.flags.writeable = False
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw face landmarks.
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        # connections = mp_face_mesh.FACEMESH_LEFT_EYE,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)    
                        
#--------------------------------------------------------------
# treating the different landmarks
                    # using my function to specifically check for the nose
                    # have the nose.csv file already ready!
                    # I want to capture the nose every thirty frames

                    
                    
                    if frame_number % 100 == 0:
                        # if frame_number == 0:
                        # I just want two dimensions, x and y, and I only have 1 coordinate (the nose)
                        #     setup_file("nose.csv", 2, 1)
                        # ! we do not need the x value
                        _, y = capture_nose(image, face_landmarks, os.path.join(abs_path, "nose.json"))
                        # _, y = capture_nose(image, face_landmarks, r"D:\Personnel\Other learning\Programming\Personal_projects\3_Hackathons_with_buddies\MAKEUOFT2024\drowsiness_detection\nose.json")

                        previous = curr
                        curr = y
                        up_or_down = curr - previous    


                    # # I want to capture the eyes every 10 frames
                    # if frame_number % 100 == 0:
                        # I want to run the setup file csv once
                        if frame_number == 0:
                            # 3 dimensions, and 12 coordinates that I am looking at
                            setup_file(os.path.join(abs_path,"eye.csv"), 3, 12)
                            setup_file(os.path.join(abs_path,"eye.csv"), 3, 478)
                        capture_eye(image, face_landmarks, os.path.join(abs_path,"eye.csv"))
                    #     if frame_number == 0:
                    #         # 3 dimensions, and 12 coordinates that I am looking at
                    #         setup_file("eye.csv", 3, 12)
                    #     capture_pose(image, face_landmarks, "pose.csv")
                    # if frame_number % 100 == 0:
                    #     if frame_number == 0:
                            
                            # 3 dimensions, and 478 coordinates that I am looking at
                        # capture_yawn(image, face_landmarks, "eye.csv")

                        # using the functions in predict.py
                        # running the prediction
                        final_prediction_eye = predict_eye(os.path.join(abs_path,"eye.csv"))
                            
                        capture_yawn(image, face_landmarks, os.path.join(abs_path,"yawn.csv"))
                        #TODO here, I run a second machine learning model that tells if the person is yawning

                        final_prediction_yawn = predict_yawn(os.path.join(abs_path,"yawn.csv"))


                        # writing the values for prediction into a json file
                        with open(os.path.join(abs_path,"predictions.json"), mode='a') as f:
                            f.truncate(0)
                            json.dump([final_prediction_eye, final_prediction_yawn, up_or_down], f)
                            print([final_prediction_eye, final_prediction_yawn, up_or_down])
#======================


            # Display the resulting frame
            cv2.imshow('MediaPipe Face Mesh', image)
            
            # stop the process
            if cv2.waitKey(5) & 0xFF == ord('q'): 
                break

            frame_number += 1

        cap.release()
        cv2.destroyAllWindows()

# if __name__ == "__main__":
open_cam()