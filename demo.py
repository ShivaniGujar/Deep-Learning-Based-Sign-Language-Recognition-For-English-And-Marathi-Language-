# -*- coding: utf-8 -*-
"""
Created on Thu May  4 13:42:32 2023

@author: Shivani Gujar
"""

import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Initialize Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Initialize logistic regression model
model = LogisticRegression()

# Set up live camera feed
cap = cv2.VideoCapture(0)

# Initialize empty lists for storing features and labels
X = []
y = []

# Start capturing and labeling live images
while True:
    ret, frame = cap.read()
    
    # Convert BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect landmarks using Mediapipe
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        results = holistic.process(image)
        
        # Extract hand landmarks
        hand_landmarks = results.right_hand_landmarks or results.left_hand_landmarks
        
        if hand_landmarks:
            # Flatten landmark points into a 1D list
            landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()
            X.append(landmarks)
            
            # Ask user for label and store in y
            label = input("Enter label (Marathi sign): ")
            y.append(label)
            
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Convert X and y to numpy arrays
X = np.array(X)
y = np.array(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model.fit(X_train, y_train)

# Evaluate model on test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save model as .pkl file
with open('marathi_sign_language_model.pkl', 'wb') as f:
    pickle.dump(model, f)
