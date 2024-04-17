'''
This file contains the augmentation process as in 03_data_preprocessing.
The processed landmarks are very heavy and take several minutes to save and load.
Producing them is faster, and should be made in 05_model_evaluation prior training
'''

import numpy as np
from collections import Counter

POSE = np.hstack((np.ones(33), np.zeros(21+21+468))) == 1
LH = np.hstack((np.zeros(33), np.ones(21), np.zeros(21+468))) == 1
RH = np.hstack((np.zeros(33+21), np.ones(21), np.zeros(468))) == 1
FACE = np.hstack((np.zeros(33+21+21), np.ones(468))) == 1
POSE_GROUPS = {
    "eyes":                 [8, 6, 5, 4, 0, 1, 2, 3, 7],   
    "mouth":                [10, 9],                       
    "right elbow":          [13],
    "right arm":            [11, 15, 17, 19, 15, 21], 
    "right body side":      [11, 23, 25, 27, 29, 31, 27],  
    "left elbow":           [14],  
    "left arm":             [12, 16, 18, 20, 16, 22],  
    "left body side":       [12, 24, 26, 28, 30, 32, 28],  
    "shoulder":             [11, 12],                      
    "waist":                [23, 24],                      
}
def pose_groups(body_parts = []):
    mask = np.zeros(len(POSE)) == 1
    for part in body_parts:
        mask[POSE_GROUPS[part]] = True
    return mask

def shift_hands(id, video,shift_rate = [[-5,5],[-5,5],[0,1]]):
    '''
    Shifts the hands, with the wrists on x,y, and z axis. Make sure the transition is smooth.
    Input: shift_rate - array of 3 tuples for x,y,z, each tuple has the shift percent for the first frame and for the last frame.
    '''
    shift_rate = np.array(shift_rate)
    if shift_rate.shape != (3,2):
        return id, video
    v = video.copy()
    interval = np.array([video[0,POSE,i].max()-video[0,POSE,i].min() for i in range(3)]) # distance left-right, up-down, depth, based on pose of first frame
    initial_shift = shift_rate[:,0] * interval / 100
    shift_step = (shift_rate[:,1] - shift_rate[:,0])*interval/(100*len(v))  # Each frame is sifted "step" percent more towards final shift
    for frame, landmarks in enumerate(v):
        shift = initial_shift + shift_step * frame
        for i in range(3): # Shift for each axis
            landmarks[POSE_GROUPS['right arm']+POSE_GROUPS['left arm'],i] += shift[i]
            landmarks[LH][:,i] += shift[i]
            landmarks[RH][:,i] += shift[i]
    new_id = id+'_shift_hands'+'_'.join([str(item) for sublist in shift_rate for item in sublist])
    return new_id, v

def flip_hands(id, video):
    v = video.copy()
    center = video[0,POSE_GROUPS['eyes'],0].mean()
    for frame in range(v.shape[0]):
        # Flip landmarks along the x-axis
        v[frame,:, 0] = 2 * center - v[frame,:, 0]

    new_id = id+'_flip'
    return new_id, v

def random_shift(seed):
    np.random.seed(seed)
    r = (np.random.rand(3, 2,)+1)*[[6,-6],[4,-4],[1.5,-1.5]] # Add 1 to prevent too small shift, then set range for each axis
    if np.random.randint(2): # No always to shift from positive to negative
        r = np.flip(r, axis=1)
    return r

# Include raw landmarks and produced augmentations
def produce_augmentations(landmarks_raw,data_info):
    seed = 1
    landmarks = landmarks_raw.copy()
    videos_per_word = Counter(data_info.word)
    for word in list(videos_per_word.keys()):
        video_ids = data_info.loc[data_info.word==word,'video_id']
        for id in video_ids:
            if id not in landmarks.keys():
                continue
            video = landmarks[id]
            landmarks[id] = video

            shift_rate = random_shift(seed) # should make some random number with a good range
            shifted_id, shifted_landmarks = shift_hands(id = id, video = video, shift_rate = shift_rate)

            mirrored_id, mirrored_landmarks = flip_hands(id = id,video = video)
            mirrored_shifted_id, mirrored_shifted_landmarks = flip_hands(id = shifted_id, video = shifted_landmarks)

            landmarks[shifted_id] = shifted_landmarks
            landmarks[mirrored_id] = mirrored_landmarks
            landmarks[mirrored_shifted_id] = mirrored_shifted_landmarks

            if videos_per_word[word] < 20: # If too few videos, produce even more
                seed += 1
                shift_rate = random_shift(seed)
                shifted_id, shifted_landmarks = shift_hands(id = id, video = video, shift_rate = shift_rate)
                landmarks[shifted_id] = shifted_landmarks
        seed += 1
    return landmarks

