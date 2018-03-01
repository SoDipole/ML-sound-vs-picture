import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def parse_sounds(path):
    raw_sounds = []
    labels = []
    for filename in os.listdir(path):
        if filename.startswith(""):
            print(filename)
            X, sample_rate = librosa.load(path+filename)
            mfccs,chroma,mel,contrast,tonnetz = extract_feature(X, sample_rate)
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])            
            
            label = filename.split(".")[0].split("_")[2]
            raw_sounds.append(np.array(X, dtype=float))
            labels.append(label)
    return raw_sounds, labels

def extract_feature(X, sample_rate):
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def plot_waves(sound_names,raw_sounds):
    for n,f in zip(sound_names,raw_sounds):
        librosa.display.waveplot(f)
        plt.title(n)
        plt.show()

path_1 = "data/test_mic/"
path_2 = "data/test_phone/"

train_features, train_labels = parse_sounds(path_1)
test_features, test_labels = parse_sounds(path_2)

clf = RandomForestClassifier(n_jobs=2, random_state=0)

clf.fit(raw_sounds, labels)

print(clf.feature_importances_)

print(clf.predict(test_sounds))
print(test_labels)

#y = raw_sounds[5]

#plt.plot(raw_sounds[5], label='пять')
#plt.show()

#D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
#librosa.display.specshow(D, y_axis='linear')
#plt.colorbar(format='%+2.0f dB')
#plt.title('Linear-frequency power spectrogram')
#plt.show()