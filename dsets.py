import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision import transforms

#torchaudio.set_audio_backend(backend="soundfile")
class ESC50Data_gpu(Dataset):
    def __init__(self,
                 path="E:\Data\ESC-50-master\ESC-50-master",
                 audio_column='filename',
                 fold_column='fold',
                 label_column='target',
                 fold=1,
                 isValSet_bool=None,
                 transform=None,
                 device=torch.device('cuda')):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        csv_path = path + "/meta/esc50.csv"
        root_dir = path + "/audio/"
        self.df = pd.read_csv(csv_path)
        self.columns_df = [audio_column, label_column]
        self.root_dir = root_dir
        self.transform = transform
        self.device = device

        # use the folds
        if isValSet_bool:
            assert fold in [1, 2, 3, 4, 5]
            self.df = self.df[self.df[fold_column] == fold]
            assert not self.df.empty, "DataFrame is empty"

        else:
            # del self.df[self.df[fold_column] == fold]
            self.df = self.df[self.df[fold_column] != fold]
            assert not self.df.empty, "DataFrame is empty"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        audio_name = os.path.join(self.root_dir, self.df.iloc[idx][self.columns_df[0]])
        waveform, sample_rate = torchaudio.load(audio_name)
        waveform = waveform.to(self.device)
        assert waveform.is_cuda
        assert waveform.sum() != 0, "the tensor is 0"

        assert self.transform is not None, "There is no transform object."

        feature_melspec_db = self.transform(waveform)

        label = self.df.iloc[idx][self.columns_df[1]]

        return feature_melspec_db, label



