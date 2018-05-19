import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def parse_sounds(path):
    #features, labels = np.empty((0,193)), np.empty(0)
    features, labels = np.empty((0,187)), np.empty(0)
    for filename in os.listdir(path):
        if filename.startswith(""):
            print(filename)
            X, sample_rate = librosa.load(path+filename)
            
            #plot_waves(X)            
            
            #S = librosa.feature.melspectrogram(y=X, sr=sample_rate)
            #librosa.display.specshow(librosa.power_to_db(S, ref=np.max), 
                                     #y_axis='mel', x_axis='time')
            #plt.title('Mel-scaled рpower spectrogram')
            #plt.show()            
            
            mfccs,chroma,mel,contrast = extract_feature(X, sample_rate)
            
            ext_features = np.hstack([mfccs,chroma,mel,contrast])
            #print(len(ext_features))
            features = np.vstack([features,ext_features])            
            
            label = filename.split(".")[0].split("_")[1]
            labels = np.append(labels, label)
            
    return np.array(features), np.array(labels, dtype = np.int)

def extract_feature(X, sample_rate):
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,
                    axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,
                     axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,
                  axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,
                       axis=0)
    return mfccs,chroma,mel,contrast

def plot_waves(raw_sound):
    librosa.display.waveplot(raw_sound)
    plt.show()

path_1 = "data/ready/train/"
path_2 = "data/ready/test/"

#train_features, train_labels = parse_sounds("data/ready/")

train_features, train_labels = parse_sounds(path_1)
test_features, test_labels = parse_sounds(path_2)

clf = KNeighborsClassifier(n_neighbors=1)

clf.fit(train_features, train_labels)

print(clf.predict(test_features))
print(test_labels)
