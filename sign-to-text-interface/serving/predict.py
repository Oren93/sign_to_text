# %%
from tensorflow.keras.models import load_model
import numpy as np
import os
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import cv2

FRAMES = 24
POSE = np.hstack((np.ones(33), np.zeros(21+21+468))) == 1
LH = np.hstack((np.zeros(33), np.ones(21), np.zeros(21+468))) == 1
RH = np.hstack((np.zeros(33+21), np.ones(21), np.zeros(468))) == 1
FACE = np.hstack((np.zeros(33+21+21), np.ones(468))) == 1
DIMENSTIONS = 3

# The following is hardcoded, will be loaded from file in later versions:
words = ['tall', 'man', 'red', 'shirt', 'play', 'basketball', 'well', 
    'neighborhood', 'cold', 'pizza', 'top', 'extra', 'cheese', 
    'taste', 'absolutely', 'delicious', 'lazy', 'Sunday',
    'afternoon', 'dark', 'room', 'lit', 'small', 'lamp', 'completely',
    'empty', 'echo', 'big', 'dog', 'wag', 'tail', 'walk',
    'beautiful', 'park', 'every', 'morning', 'short',
    'woman', 'wear', 'colorful', 'dress', 'have', 'beautiful',
    'daughter', 'excel', 'academics', 'sports', 'good']
index_to_word = {word: i for i, word in enumerate(words)}
# %%
newest_model = max(os.listdir('lstm'))
model = load_model(os.path.join('lstm',newest_model,'sign_to_text.keras'))
#models_dir = '/app/serving/lstm'
#newest_model = max(os.listdir(models_dir))
#model = load_model(os.path.join(models_dir,newest_model,'sign_to_text.keras'))

#%%
def get_landmarks(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33,3))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21,3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21,3))
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468,3))
   
    landmarks = np.vstack((pose, lh, rh, face))
    return landmarks

def extarct_landmarks(video_path):
    mp_holistic = mp.solutions.holistic # Holistic model
    cap = cv2.VideoCapture(video_path)
    # Set mediapipe model
    ret, frame = cap.read()
    video_landmarks = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            # Read feed
            ret, frame = cap.read()
            if not ret:
                break
            # Make detections
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame)  # Finds pose
            # save the landmarks for that frame
            frame = get_landmarks(results)
            video_landmarks.append(frame)
    cap.release()
    return np.stack(video_landmarks, axis=0)

def pick_frames(video,num_frames):
    if len(video) < num_frames:
        video_longer = video.copy()
        for _ in range(len(video),num_frames):
            video_longer = np.append(video_longer,video_longer[-1])
        return video_longer
    step_size = len(video) // num_frames
    video_shorter = video[::step_size][:num_frames]
    return video_shorter

# %%
def make_prediction(video):
    landmarks = extarct_landmarks(video)
    landmarks = pick_frames(landmarks[:,POSE+LH+RH,:DIMENSTIONS],FRAMES)

    # Note: the model expect shape (None, 24, 225) so we wrap the array more
    landmarks = np.array([[frame.flatten() for frame in landmarks]])

    predict_result = model.predict(landmarks)
    word_index = np.argmax(predict_result)
    predicted_word = words[word_index]
    confidence = predict_result[0,word_index]
    return predicted_word, confidence
