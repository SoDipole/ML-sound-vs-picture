import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def load_sound_files(path):
    raw_sounds = []
    for filename in os.listdir(path):
        if filename.startswith("f"):
            print(filename)
            X,sr = librosa.load(path+filename)
            raw_sounds.append(X)
    return raw_sounds

def plot_waves(sound_names,raw_sounds):
    for n,f in zip(sound_names,raw_sounds):
        librosa.display.waveplot(f)
        plt.title(n)
        plt.show()

path = "data/test_mic/"
sound_names = ["5","6","7","8","9","5","6","7","8","9"]

raw_sounds = load_sound_files(path)

plot_waves(sound_names,raw_sounds)

#y = raw_sounds[5]

#plt.plot(raw_sounds[5], label='пять')
#plt.show()

#D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
#librosa.display.specshow(D, y_axis='linear')
#plt.colorbar(format='%+2.0f dB')
#plt.title('Linear-frequency power spectrogram')
#plt.show()