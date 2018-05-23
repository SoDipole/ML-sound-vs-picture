import os
import librosa
import numpy as np

path = "data/raw/"
out_path = "data/ready/"

if not os.path.exists(out_path):
    os.makedirs(out_path)

maxv = np.iinfo(np.int16).max

person_id = 0

for filename in os.listdir(path):
    if filename.startswith(""):
        person_id += 1     
        print(filename)
        X, sample_rate = librosa.load(path+filename)
        
        chunks = librosa.effects.split(X, top_db=30, frame_length=5500)
        
        print(len(chunks))
        
        label = 0
        for chunk in chunks:
            label += 1
            y = X[chunk[0]:chunk[1]]
            librosa.output.write_wav(out_path+str(person_id)+"_"+str(label)+".wav", 
                                     (y*maxv).astype(np.int16),
                                     sample_rate)
