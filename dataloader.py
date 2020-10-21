from torch.utils.data import Dataset, DataLoader
import pandas as pd
import librosa
import os
import time
from signal_process import signal_process
import random
import math


class KeyDataset(Dataset):
    def __init__(self, metadata_path='metadata.csv', audio_dir='audio', sr=22050, split='training'):
        super(KeyDataset, self).__init__()
        
        self.metadata_path = metadata_path
        self.audio_dir = audio_dir
        self.sr = sr
        self.split = split
        self.track_column_name = 'track_id'
        self.label_column_name = 'track_key'
        
        self.metadata_df = pd.read_csv(metadata_path)
        self.label_dict = {k: v for v, k in enumerate(sorted(set(self.metadata_df[self.label_column_name].values)))}
        
        # return only split
        self.metadata_df = self.metadata_df[self.metadata_df['split'] == self.split]

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        
        track_id = self.metadata_df.iloc[idx][self.track_column_name]
        label = self.label_dict[self.metadata_df.iloc[idx][self.label_column_name]]
        audio, sr = librosa.load(os.path.join(self.audio_dir, (f"{track_id}.wav")), mono=True, sr=self.sr)                                
        # audio_len = len(audio)
        # start_point = random.randint(0, audio_len/3)
        # audio = audio[start_point : math.floor(start_point + audio_len*2/3)]
        return audio, label


if __name__ == '__main__':
    
    method = 'logmelspectrogram'

    start = time.time()
    dataset = KeyDataset(metadata_path='metadata.csv', audio_dir='audio', sr=22050, split='training')
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    for idx, x in enumerate(train_loader):
        audio, label = x
        processed_audio = signal_process(audio, sr=22050, method=method)
        print(audio.size())
    # print(method, time.time()-start)
