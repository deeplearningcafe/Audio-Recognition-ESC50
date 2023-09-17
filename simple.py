import time

import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import os
import datetime
import logging
logging.basicConfig(level=logging.DEBUG)

from torch.utils.tensorboard import SummaryWriter

# it takes 11 minutes to complete, the same as in jupyter, ends with val Accuracy: 1.0000, val AVG Loss: 0.0435
# using enumerate is takes almost the same as without using enumerate for the dataset

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.activation = nn.Tanh()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 64 * 215, 128)  # Assuming input image size is (128, 431)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = x.view(-1, 16 * 64 * 215)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x


class ESC50Data_gpu(Dataset):
    def __init__(self,
                 csv_file,
                 root_dir,
                 audio_column='filename',
                 fold_column='fold',
                 label_column='category',
                 fold=1,
                 isValSet_bool=None,
                 transform=None,
                 device=torch.device('cuda')):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.df = pd.read_csv(csv_file)
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

        if self.transform:
            feature_melspec_db = self.transform(waveform)

        label = self.df.iloc[idx][self.columns_df[1]]

        return feature_melspec_db, label

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

mean = [-8.8024]
std = [20.8701]
sample_rate = 44100

data_transform = transforms.Compose([
    torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=128, n_fft=2048, hop_length=512, f_min=0, f_max=8000).to(device),
    torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80).to(device),
    transforms.Normalize(mean, std).to(device)
])
log_dir = os.path.join('runs', "Simple",  datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S'))
writer_val = SummaryWriter(log_dir=log_dir + '-val_cls-' + "totalTrainingSamples_count")
writer_trn = SummaryWriter(log_dir=log_dir + '-trn_cls-' + "totalTrainingSamples_count")

log = logging.getLogger(__name__)

def logMetrics(fold, epoch, phase, metrics, totalTrainingSamples_count):

    if phase == 'val':
        avg_loss = 0.0
        avg_acc = 0.0
        for element in metrics:
            avg_loss += element["loss"]
            avg_acc += element["acc"]

        avg_loss = avg_loss / len(metrics)
        avg_acc = avg_acc / len(metrics)

        time = datetime.datetime.now()
        log.info(
            "Time {} Fold {} | Epoch {} | {} Loss: {:.4f} | {} Accuracy: {:.2%}".format(
                time, fold, epoch, phase, avg_loss, phase, avg_acc
            )
        )
        # Clear metrics for the next fold


        writer_val.add_scalar('Loss/Validation', avg_loss, totalTrainingSamples_count)
        writer_val.add_scalar('Accuracy/Validation', avg_acc, totalTrainingSamples_count)

    elif phase == 'train':  # For training phase
        avg_loss = 0.0
        avg_acc = 0.0
        for element in metrics:
            avg_loss += element["loss"]
            avg_acc += element["acc"]

        avg_loss = avg_loss / len(metrics)
        avg_acc = avg_acc / len(metrics)

        time = datetime.datetime.now()
        log.info(
            "Time {} Fold {} | Epoch {} | {} Loss: {:.4f} | {} Accuracy: {:.2%}".format(
                time, fold, epoch, phase, avg_loss, phase, avg_acc
            )
        )
        # Log metrics to TensorBoard
        writer_trn.add_scalar('Loss/Training', avg_loss, totalTrainingSamples_count)
        writer_trn.add_scalar('Accuracy/Training', avg_acc, totalTrainingSamples_count)


def computeAccuracy(predicted_labels, true_labels):
    correct = (predicted_labels == true_labels).sum().item()
    total = len(true_labels)
    accuracy = correct / total
    return accuracy


def cross_validation(num_folds, n_epochs, batch_size,
                     csv_file,
                     root_dir,
                     audio_column='filename',
                     fold_column='fold',
                     label_column='target',
                     isValSet_bool=None,
                     transform=None,
                     device=torch.device('cuda')):
    model = CNNModel(50).to(device=device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.99)
    loss_fn = nn.CrossEntropyLoss()
    totalTrainingSamples_count = 0

    for fold in range(num_folds):
        print(f"Time {datetime.datetime.now()}, Fold {fold + 1}/{num_folds}")
        dataset = ESC50Data_gpu(csv_path, root_dir, *columns, fold + 1, False, data_transform, device)
        dataset_val = ESC50Data_gpu(csv_path, root_dir, *columns, fold + 1, True, data_transform, device)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   shuffle=False)
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size,
                                                 shuffle=False)

        for epoch in range(1, n_epochs + 1):
            loss_train = 0.0

            # len(train_loader) is 5, 1600 / 320 = 5, and the batch size is 320, 2 secs to get the train loader
            start = time.time()
            for specs, labels in train_loader:
                elapsed_time = time.time() - start
                print("time to get train loader", elapsed_time)
                model.train()
                specs = specs.to(device=device)
                labels = labels.to(device=device)
                outputs = model(specs)
                loss = loss_fn(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_train += loss.item()
                totalTrainingSamples_count += len(train_loader.dataset)

            if epoch == 1 or epoch % 5 == 0:
                model.eval()
                for name, loader in [("train", train_loader), ("val", val_loader)]:
                    correct = 0
                    total = 0
                    val_loss = 0.0
                    fold_metrics = []

                    with torch.no_grad():  # <1>
                        for imgs, labels in loader:
                            imgs = imgs.to(device)
                            labels = labels.to(device)
                            outputs = model(imgs)
                            loss = loss_fn(outputs, labels)
                            val_loss += loss.item()

                            _, predicted = torch.max(outputs, dim=1)  # <2>
                            total += labels.shape[0]  # <3>
                            correct += int((predicted == labels).sum())  # <4>
                            acc = computeAccuracy(predicted, labels)
                            metrics_loop = {
                                'loss': loss.item(),
                                'acc': acc,
                            }
                            fold_metrics.append(metrics_loop)

                    logMetrics(fold, epoch, name, fold_metrics, totalTrainingSamples_count)
                    """accuracy = correct / total
                    avg_val_loss = val_loss / len(val_loader)
                    avg_train_loss = loss_train / len(train_loader)
                    # print("Accuracy {}: {:.2f}".format(name , accuracy))
                    print(
                        f"Time {datetime.datetime.now()}, Epoch {epoch + 1}/{n_epochs}, AVG Training loss: {avg_train_loss:.4f}, {name} Accuracy: {accuracy:.4f}, {name} AVG Loss: {avg_val_loss:.4f}")
                    """


path = "E:\Data\ESC-50-master\ESC-50-master"

batch_size = 320
csv_path = path + "/meta/esc50.csv"
root_dir = path + "/audio/"
columns = ['filename', 'fold', 'target']
args = [csv_path, root_dir]

cross_validation(num_folds=5,
    n_epochs = 10,
    batch_size=320,
    csv_file=csv_path,
    root_dir=root_dir)
