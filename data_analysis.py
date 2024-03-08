
# %% Imports
import os
import json
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% Loading Data into CSV File (with removal of missing files) 

# Collect info about video files and store in panda dataframe
def video_files_to_pandas():
    stats = pd.DataFrame({'video_id': [], 'fps':[],'width':[],'height':[], 'duration':[]})
    for i, file in enumerate(os.listdir('videos')):
        video_file = os.path.join('videos',file)
        video = cv2.VideoCapture(video_file)
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        if int(major_ver)  < 3 :
            frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
            print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
        else :
            frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)
        video_read = video.read()
        if video_read[0]:
            h, w, _ = video_read[1].shape
        stats.loc[i] = [file.replace('.mp4',''), fps, w, h, frames/fps]
        video.release()
    return stats

# Converts Json file into panda format (excluding missing)
def json_to_pandas():
    with open("data/missing.txt", 'r') as file:
        missing = [int(line.strip()) for line in file]

    f = open("data/WLASL_v0.3.json")
    data = json.load(f)
    f.close()

    columns = [
        'bbox',
        'fps',
        'frame_end',
        'frame_start',
        'instance_id',
        'signer_id',
        'source',
        'split',
        'url',
        'variation_id',
        'video_id',
        'word'
        ]
    
    data_info = pd.DataFrame(columns=columns)
    for i, word in enumerate(data):
        for instance in word['instances']:
            if int(instance['video_id']) in missing:
                continue
            instance['word'] = word['gloss']
            data_info.loc[instance['video_id']] = instance
    return data_info

# Combines video files in panda format and json file in panda format
def load_data(filename):
    if filename in os.listdir():
        data = pd.read_csv('data/video_labels.csv',dtype={'video_id':object})
    else:
        video_pandas = video_files_to_pandas()
        json_pandas = json_to_pandas()
        
        json_pandas['video_id'] = pd.to_numeric(json_pandas['video_id'])
        video_pandas['video_id'] = pd.to_numeric(video_pandas['video_id'])
        data = json_pandas.merge(video_pandas, on='video_id', how='left')
        data.to_csv('video_labels.csv')
    return data

data = load_data("video_labels.csv")
# %% Data Analysis
def analysis_of_data_count(data):

    # Data Count for Full Dataset
    print(f"Number of Unique Words: {data['word'].nunique()}")
    print(f"Number of Unique Sources: {data['source'].nunique()}")
    print(f"Number of Unique Signers: {data['signer_id'].nunique()}")

    print("\n")
    print(f"Total Instances (complete data): {len(data)}")
    print(f"Total Train Instances (complete data): {len(data[data['split'] == 'train'])}")
    print(f"Total Test Instances (complete data): {len(data[data['split'] == 'test'])}")
    print(f"Total Test Instances (complete data): {len(data[data['split'] == 'val'])}")

    value_counts_complete = data['word'].value_counts()
    print(f"Average Instances per word: {value_counts_complete.mean()}")
    print(f"Minimum Instances per word: {value_counts_complete.min()}")
    print(f"Maximum Instances per word: {value_counts_complete.max()}")

    print("\n")
    print(f"Total Instances Missing: {len(data[data['missing'] == True])}")
    print(f"Total Instances Present: {len(data[data['missing'] == False])}")

    # Count of data present after removal
    data_present = data[data['missing'] == False]

    value_counts_train = (data_present[data_present['split'] == 'train'])['word'].value_counts()
    value_counts_test = (data_present[data_present['split'] == 'test'])['word'].value_counts()
    value_counts_val = (data_present[data_present['split'] == 'val'])['word'].value_counts()

    print("\n")
    print("With data available:")
    print(f"Number of Unique Words: {data_present['word'].nunique()}")
    print(f"Number of Unique Sources: {data_present['source'].nunique()}")
    print(f"Sources: {data_present['source'].unique()}")
    print(f"Number of Unique Signers: {data_present['signer_id'].nunique()}")

    value_counts_present = data_present['word'].value_counts()
    print(f"Average Instances per word: {value_counts_present.mean()}")
    print(f"Minimum Instances per word: {value_counts_present.min()}")
    print(f"Maximum Instances per word: {value_counts_present.max()}")

    print("\n")
    print(f"Total Instances: {len(data_present)}")

    print("\n")
    print(f"Total Train Instances: {len(data_present[data_present['split'] == 'train'])}")
    print(f"Number of Unique Sources: {(data_present[data_present['split'] == 'train'])['source'].nunique()}")
    print(f"Number of Unique Signers: {(data_present[data_present['split'] == 'train'])['signer_id'].nunique()}")
    print(f"Average Train Instances per word: {value_counts_train.mean()}")
    print(f"Minimum Train Instances per word: {value_counts_train.min()}")
    print(f"Maximum Train Instances per word: {value_counts_train.max()}")

    print("\n")
    print(f"Total Test Instances: {len(data_present[data_present['split'] == 'test'])}")
    print(f"Number of Unique Sources: {(data_present[data_present['split'] == 'test'])['source'].nunique()}")
    print(f"Number of Unique Signers: {(data_present[data_present['split'] == 'test'])['signer_id'].nunique()}")
    print(f"Average Test Instances per word: {value_counts_test.mean()}")
    print(f"Minimum Test Instances per word: {value_counts_test.min()}")
    print(f"Maximum Test Instances per word: {value_counts_test.max()}")

    print("\n")
    print(f"Total Validation Instances: {len(data_present[data_present['split'] == 'val'])}")
    print(f"Number of Unique Sources: {(data_present[data_present['split'] == 'val'])['source'].nunique()}")
    print(f"Number of Unique Signers: {(data_present[data_present['split'] == 'val'])['signer_id'].nunique()}")
    print(f"Average Validation Instances per word: {value_counts_val.mean()}")
    print(f"Minimum Validation Instances per word: {value_counts_val.min()}")
    print(f"Maximum Validation Instances per word: {value_counts_val.max()}")


analysis_of_data_count(data)

# %% Data Analysis
# Expensive Operation
def calculate_brightness_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)
    brightness = np.mean(gray)
    return brightness, contrast

def calculate_video_contrast(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()
    ret, frame = cap.read()
    contrasts = []
    brightnesses = []

    while ret:
        brightness, contrast = calculate_brightness_contrast(frame)
        contrasts.append(contrast)
        brightnesses.append(brightness)
        ret, frame = cap.read()

    cap.release()
    average_brightness = np.mean(brightnesses)
    std_brightness = np.std(brightnesses)
    average_contrast = np.mean(contrasts)
    std_contrast = np.std(contrasts)
    return average_brightness, std_brightness, average_contrast, std_contrast

def calculate_brightness_contrast_for_all():
    data = pd.DataFrame({'video_id': [], 'average brightness':[],'std brightness':[],'average contrast':[], 'std contrast':[]})
    for i, file in enumerate(os.listdir('videos')):
        video_file = os.path.join('videos', file)
        average_contrast, std_contrast = calculate_video_contrast(video_file)
        average_brightness, std_brightness, average_contrast, std_contrast.append(average_contrast)
        std_contrast_arr.append(std_contrast)


calculate_video_contrast("videos/00336.mp4")

# %% Data Analysis

def analysis_of_video(data):
    data_present = data[data['missing'] == False]

    print("\n")
    print(f"Total Length of all videos: {data_present['duration'].sum()}")
    print(f"Average Length of all videos: {data_present['duration'].mean()}")
    print(f"Minumum Length of all videos: {data_present['duration'].min()}")
    print(f"Maximum Length of all videos: {data_present['duration'].max()}")
    print(f"Average Length of videos in Train: {(data_present[data_present['split'] == 'train'])['duration'].mean()}")
    print(f"Average Length of videos in Test: {(data_present[data_present['split'] == 'test'])['duration'].mean()}")
    print(f"Average Length of videos in Validation: {(data_present[data_present['split'] == 'val'])['duration'].mean()}")


    print("\n")
    data_total_pixel = data_present['width'] * data_present['height']
    data_train_pixel = (data_present[data_present['split'] == 'train'])['width'] * (data_present[data_present['split'] == 'train'])['height']
    data_test_pixel = (data_present[data_present['split'] == 'test'])['width'] * (data_present[data_present['split'] == 'test'])['height']
    data_val_pixel = (data_present[data_present['split'] == 'val'])['width'] * (data_present[data_present['split'] == 'val'])['height']
    plt.boxplot([data_total_pixel, data_train_pixel, data_test_pixel, data_val_pixel], 
                labels=['Complete Set', 'Training Set', 'Test Set', 'Validation Set'])
    plt.xlabel('Data Sets')
    plt.ylabel('Pixel Count / Resolution (derived from Height * Width)')
    plt.show()

    #average_contrast_arr = []
    #std_contrast_arr = []
    #for i, file in enumerate(os.listdir('videos')):
        #video_file = os.path.join('videos', file)
        #average_contrast, std_contrast = calculate_video_contrast(video_file)
        #average_contrast_arr.append(average_contrast)
        #std_contrast_arr.append(std_contrast)
    #print(f"Average standard deviation in contrast in each video: {(pd.Series(std_contrast_arr)).mean()}")
    #print(f"Average contrast in each video: {(pd.Series(average_contrast_arr)).mean()}")
    #plt.boxplot(average_contrast_arr)
    #plt.ylabel('Contrast')
    #plt.show()




analysis_of_video(data)
# %% Vehn diagram
signer_ids_train = data.loc[data.split=='train','signer_id'].unique()
signer_ids_val = data.loc[data.split=='val','signer_id'].unique()
signer_ids_test = data.loc[data.split=='test','signer_id'].unique()

print('train only -',len(np.setdiff1d(signer_ids_train,  np.append(signer_ids_val,signer_ids_test))))
print('val only - ',len(np.setdiff1d(signer_ids_val, np.append(signer_ids_test,signer_ids_train))))
print('test only - ',len(np.setdiff1d(signer_ids_test, np.append(signer_ids_val,signer_ids_train))))

print('intersect all',len(np.intersect1d(signer_ids_test, np.intersect1d(signer_ids_val,signer_ids_train))))

print('intersetc train val',len(np.intersect1d(signer_ids_train, np.setdiff1d(signer_ids_val,signer_ids_test))))
print('intersetc train test',len(np.intersect1d(signer_ids_train, np.setdiff1d(signer_ids_test,signer_ids_val))))
print('intersetc test val',len(np.intersect1d(signer_ids_test, np.setdiff1d(signer_ids_val,signer_ids_train))))


# %% Analysis on word distribution accross the splits
word_stats = pd.DataFrame({'word':data.word.unique(),'appearances':0,'signers':0,'train':0,'val':0,'test':0})
for word in data.word.unique():
    word_subset = data.loc[data.word==word,:]
    row = {
        'word': word,
        'appearances': len(word_subset),
        'signers':  str(word_subset.signer_id.unique()),
        'train':    len(word_subset.loc[word_subset.split=='train']),
        'val':  len(word_subset.loc[word_subset.split=='val']),
        'test': len(word_subset.loc[word_subset.split=='test'])
        }
    
    word_stats.loc[word_stats.word==word] = row.values()
word_stats
# %%