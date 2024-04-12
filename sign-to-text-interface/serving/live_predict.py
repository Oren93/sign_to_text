# %%
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2

# The following is hardcoded, will be loaded from file in later versions:
words = ["clothes","help","book","chair","before","computer","deaf",
         "thin","fine","drink","candy","go","no","cousin","walk","who",]

index_to_word = {word: i for i, word in enumerate(words)}
# %%
model_version = '4'
RESOLUTION = (256,256) # model no 4 accept this input size

model = load_model(os.path.join('cnn',model_version,'sign_to_text.keras'))

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
    cap = cv2.VideoCapture(video)
    index = 0
    sentence = []
    word_probabilities = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        index += 1
        if index < starting_frame:
            continue
        if index % proccess_rate == 0:
            resized_frame = cv2.resize(frame, RESOLUTION)
            probabilities = model.predict(resized_frame[np.newaxis, ...],verbose =0)
            argmax = probabilities.argmax(axis=1)[0] # most likely prediction

            sentence.append(words[argmax])
            word_probabilities.append(probabilities[0,argmax])

    starting_frame = index
    cap.release()

    return index, collect_consecutive_entries(sentence, word_probabilities)
