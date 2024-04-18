# %%
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2
import mediapipe as mp
from words import words_for_model

POSE = np.hstack((np.ones(33), np.zeros(21+21+468))) == 1
LH = np.hstack((np.zeros(33), np.ones(21), np.zeros(21+468))) == 1
RH = np.hstack((np.zeros(33+21), np.ones(21), np.zeros(468))) == 1
FACE = np.hstack((np.zeros(33+21+21), np.ones(468))) == 1
# %%
model_version = '7'
words = words_for_model[model_version]
index_to_word = {word: i for i, word in enumerate(words)}

model = load_model(os.path.join('landmark_nn',model_version,'sign_to_text.keras'))
#%% Get landmarks
def get_landmarks(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33,3))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21,3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21,3))
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468,3))

    landmarks = np.vstack((pose, lh, rh, face))
    return landmarks
# %%
def collect_consecutive_entries(sentence, word_probabilities):
    result = []
    current_entry = None
    max_probability = float('-inf')

    for i, (word, probability) in enumerate(zip(sentence, word_probabilities)):
        if word == current_entry:
            if probability > max_probability:
                max_probability = probability
        else:
            if current_entry is not None:
                result.append((current_entry, max_probability))
            current_entry = word
            max_probability = probability

    # Handle the last entry
    if current_entry is not None:
        result.append((current_entry, max_probability))

    return np.array(result)

def make_live_prediction(video,proccess_rate,starting_frame=0):
    '''
    video: the path for the video file
    proccess_rate: 1 frame out of how many frame will be processed
    starting_frame: the video is being sent as a whole as it is being recorded, we need to
                    monitor where we stopped and continue from there.
    '''
    mp_holistic = mp.solutions.holistic # Holistic model
    cap = cv2.VideoCapture(video)
    index = 0
    sentence = []
    word_probabilities = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            index += 1
            if index < starting_frame:
                continue
            if index % proccess_rate == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame)
                landmarks = get_landmarks(results)
                if np.sum(landmarks[LH+RH]) == 0: # Hands out of frame
                    continue
                landmarks = landmarks.flatten()
                probabilities = model.predict(landmarks[np.newaxis, ...] ,verbose =0)
                argmax = probabilities.argmax(axis=1)[0] # most likely prediction

                sentence.append(words[argmax])
                word_probabilities.append(probabilities[0,argmax])

    starting_frame = index
    cap.release()

    return index, collect_consecutive_entries(sentence, word_probabilities)
