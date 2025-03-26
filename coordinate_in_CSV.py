# -*- coding: utf-8 -*-
"""
To add new marathi sign in coords1.csv
"""

import csv
import os
import numpy as np#to perform numerical operations

import mediapipe as mp
import cv2#to open a live camera

#Initialize Mediapipe's drawing and holistic solutions
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

#Define the number of landmarks to detect for each hand
num_coords = 21
print(num_coords)

#Create a list of column names for the CSV file that will be created
landmarks = ['class']
for val in range(1, num_coords+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
print(landmarks)

#Create a new CSV file with the given column names
with open('coords1.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)
  
  #Define the name of the class being tracked
class_name = "Chann"

# Open the default camera to capture video
cap = cv2.VideoCapture(0)

# Initiate holistic model for detecting hand landmarks
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # Loop through each frame of the webcam feed    
    while cap.isOpened():
        ret, frame = cap.read()# Read each frame from the webcam feed
        
        # Recolor Feed img to RGB format
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # Disable writing to image memory to improve performance      
        
        # # Make Detections using Mediapipe's holistic solution
        results = holistic.process(image)
        # 
        
        # 
        # Enable writing to image memory
       
        image.flags.writeable = True  
        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #                          
        # 3. Draw the landmarks on the left hand in the image
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                  )

        # # 
        # Export coordinates to a CSV file
        try:
            # Extract Pose landmarks from result
            left = results.left_hand_landmarks.landmark
            # Create a list of coordinates for each landmark and flatten it into a 1D array
            left_hand = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left]).flatten())
            
            # 
            #     # Combine the class name with the detected landmarks
            row = left_hand
            row.insert(0, class_name)
            
            #  # Append the detected landmarks to the CSV fil
            with open('coords1.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row) 
            
        except:
            pass
                        
        cv2.imshow('Raw Webcam Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()

