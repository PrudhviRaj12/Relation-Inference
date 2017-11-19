
# coding: utf-8

# In[15]:

import os
import operator



current_dir = os.getcwd()
#JE_files = os.listdir('C:\\Users\\prudh\\AudioData\\JE')

#os.chdir('C:\\Users\\prudh\\AudioData\\JE')

import librosa
import re

def readfiles(filenames):
    c = re.compile('.wav')
    sounds = dict()
    for filename in filenames:
        ct = re.sub(c, '', filename)
        X, sr = librosa.load(filename)
        sounds[ct] = X
    return sounds

#peaker = ['DC', 'KL', 'JK', 'JE']

#or s in speaker:
#JE_files = os.listdir(current_dir + '\\' + 'JE')
#os.chdir(current_dir + '\\' + 'JE')
#all_files_JE = readfiles(JE_files)
speaker = ['DC', 'JE', 'JK', 'KL']
all_files_JE = []
for s in speaker:
    #print current_dir + '\\' + s
    os.chdir(current_dir + '\\' + s)
    #print os.listdir(os.getcwd())
    all_files_JE = readfiles(os.listdir(os.getcwd()))
    #print all_files_JE

#print len(all_files_JE)
def encode_targets(all_files_JE, targ_loc, targ_str, value):
    for a in all_files_JE:
        #print a
        #print all_files_JE
        #print a[targ_loc]
        if a[targ_loc] == targ_str:
            all_files_JE[a] = all_files_JE[a].tolist()
            all_files_JE[a].append(value)
    return all_files_JE

def target_attacher(all_files_JE):
    all_files_JE_with_targets = dict()
# Encoding Anger
    all_files_JE_with_targets = encode_targets(all_files_JE, 0, 'a', 1)
# Encoding Disgust
    all_files_JE_with_targets = encode_targets(all_files_JE, 0, 'd', 2)
# Encoding Fear
    all_files_JE_with_targets = encode_targets(all_files_JE, 0, 'f', 3)
#Encoding Happiness
    all_files_JE_with_targets = encode_targets(all_files_JE, 0, 'h', 4)
# Encoding Neutral
    all_files_JE_with_targets = encode_targets(all_files_JE, 0, 'n', 5)
# Encoding Sadness
    all_files_JE_with_targets = encode_targets(all_files_JE, 1, 'a', 6)
# Encoding Surprise
    all_files_JE_with_targets = encode_targets(all_files_JE, 1, 'u', 7)
    
    return all_files_JE_with_targets

def feature_and_targets(all_files_JE):
    all_files_JE_with_targets = target_attacher(all_files_JE)
    feature_values, target_values = [], []
    for a in all_files_JE_with_targets:
        range_len = len(all_files_JE_with_targets[a]) - 1
        feature_values.append(all_files_JE_with_targets[a][0 : range_len])
        target_values.append(all_files_JE_with_targets[a][-1])
        
    return feature_values, target_values

feature_values, target_values = feature_and_targets(all_files_JE)

import numpy as np

def compute_mfcc(data):
    mfccs = []
    for d in data:
        d = np.array(d)
        mfccs.append(np.mean(librosa.feature.mfcc(y = d, sr = 22050, n_mfcc= 40).T, axis=0))
    return mfccs

def find_indexes_of_emotion(target_values, emotion):
    index_list = []
    for t in range(0, len(target_values)):
        if target_values[t] == emotion:
            index_list.append(t)
    return index_list

def collect_files(target_values, emotion):
    indexes = find_indexes_of_emotion(target_values, emotion)
    #print len(indexes)
    #print len(feature_values)
    files = []
    for i in indexes:
        files.append(feature_values[i])
    return files

def collect_all_files(feature_values, target_values):
    emotions = np.unique(target_values)
    anger = collect_files(target_values, emotions[0])
    disgust = collect_files(target_values, emotions[1])
    fear = collect_files(target_values, emotions[2])
    happy = collect_files(target_values, emotions[3])
    neutral = collect_files(target_values, emotions[4])
    sadness = collect_files(target_values, emotions[5])
    surprise = collect_files(target_values, emotions[6])
    
    total = anger + disgust + fear + happy + neutral + sadness + surprise
    
    total_target = [emotions[0]] * len(anger) + [emotions[1]] * len(disgust) + [emotions[2]] * len(fear) +     [emotions[3]] * len(happy) + [emotions[4]] * len(neutral) + [emotions[5]] * len(sadness) +     [emotions[6]] * len(surprise)
            
    return total, total_target

def enumerate_emotions(predicted_emotion):
    enum_emotions = {1 : 'Anger', 2 : 'Disgust', 3 : 'Fear', 4 : 'Happiness',
                    5 : 'Neutral', 6: 'Sadness', 7 : 'Surprise'}
    return enum_emotions[predicted_emotion[0]]

speaker = ['DC', 'JE', 'JK', 'KL']

def collect_speaker_files(speaker, current_dir):
    change_dir = current_dir + '\\' + speaker
    os.chdir(change_dir)
    speaker_files = readfiles(os.listdir(change_dir))
    features, targets = feature_and_targets(speaker_files)
    total_files, total_targets = collect_all_files(features, targets)
    
    return total_files, total_targets

def all_files(speaker, current_dir):
    features, targets = [], []
    for s in speaker:
        total_files, total_targets = collect_speaker_files(s, current_dir)
        features += total_files
        targets += total_targets
    return features, targets

all_features, all_targets = all_files(speaker, current_dir)


import pickle
os.chdir(current_dir)
name = open('all_features.txt', 'wb')
pickle.dump(all_features, name)
name.close()

name = open('all_targets.txt', 'wb')
pickle.dump(all_targets, name)
name.close()

name = open('all_features.txt', 'rb')
all_features = pickle.load(name)
name.close()

name = open('all_targets.txt', 'rb')
all_targets = pickle.load(name)
name.close()
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
lr = LogisticRegression(penalty= 'l2', C= 0.9)
lr.fit(compute_mfcc(all_features), all_targets)

import os
current_dir = os.getcwd()
import librosa
import operator
argument = 'signal'
if argument == 'signal':
    os.chdir(current_dir)
    x, sr = librosa.load(filename)
    c = lr.predict_log_proba(compute_mfcc([x]))
    lr.predict(compute_mfcc([x]))
    dict_new = dict()
    for i in range(1, 8):
        dict_new[i] = c[0][i-1]
    sorted_x = sorted(dict_new.items(), key=operator.itemgetter(0))
    v = (sorted(dict_new.items(), key=lambda x: x[1], reverse= True))
    pred_emotions = []
    for ve in v[0:3]:
        pred_emotions.append(enumerate_emotions(ve))

    print ' Top Three Emotions : 
    print pred_emotions

argument = 'conversation'
if argument == 'conversation':
    current_conv = os.listdir(current_dir + '\\' + 'Conversations' + '\\' + conv_dir)
    os.chdir(current_dir + '\\' + 'Drink To That')
    v_files = []
    for v in current_conv:
        v_files.append(librosa.load(v)[0])
    pred_emotions, pred_list = [], []
    for v in v_files:
        pred_emotions.append((lr.predict(compute_mfcc([v]))).tolist())
    for pred in pred_emotions:
        pred_list.append(pred[0])
    unique =np.unique(pred_list)
    if len(unique) > 2:
        print 'Number of Emotions in this Conversation : ' + str(len(unique))
        print 'Personal'
    else:
        print 'Number of Emotions in this Conversation: ' + str(len(unique))
        print 'Professional'
   

