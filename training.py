import argparse
import datetime
import os
import random
import sys

import numpy as np

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, ConcatDataset

from model import CNNModel, CNNModel_2, CNNModel_BatchNorm, CNNModel_Res, CNNModel_ResNet, CNNModel_AlexNet, \
    CNNModel_ResBlocks, CNNModel_BaseModel, CNNModel_BaseModelV3, CNNModel_BaseModelV4, CNNModel_BaseModelV5
from dsets import ESC50Data_gpu
import torchaudio
from torchvision import transforms
import logging
import shutil

logging.basicConfig(level=logging.DEBUG)

log = logging.getLogger(__name__)


# 乱数シードを設定
seed = 42
torch.manual_seed(seed)  # PyTorchの乱数シードを設定
random.seed(seed)  # Pythonの標準の乱数シードを設定
np.random.seed(seed)  # NumPyの乱数シードを設定

# GPUが利用可能な場合、その乱数シードも設定
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# it takes 2.24 to get the data loader
class TrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
                            help='Batch size to use for training',
                            default=400,
                            type=int,
                            )
        parser.add_argument('--epochs',
                            help='Number of epochs to train for',
                            default=3,
                            type=int,
                            )

        parser.add_argument('--tb-prefix',
                            default='CNN',
                            help="Data prefix to use for Tensorboard run.",
                            )

        parser.add_argument('comment',
                            help="Comment suffix for Tensorboard run.",
                            nargs='?',
                            default='CNNModel_BaseModelV5_TestWriters_Folds_Augmented_8000',
                            )
        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.totalTrainingSamples_count = 0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()
        self.transform = self.initTransform()
        self.augmentedtransform = self.initAugmentedTransform(self.transform)

        self.path = "E:\Data\ESC-50-master\ESC-50-master"
        self.columns = ['filename', 'fold', 'target']
        self.num_folds = 5


        self.writer_val = self.initTensorboardWriters("val")
        self.writer_trn = self.initTensorboardWriters("train")




    def initModel(self):
        #model = CNNModel_2(50)
        #model = CNNModel(50)
        #model = CNNModel_BatchNorm(50)
        #model = CNNModel_Res(50)
        #model = CNNModel_ResNet(50)
        #model = CNNModel_AlexNet(50)
        #model = CNNModel_ResBlocks(50)
        #model = CNNModel_BaseModel(50)
        #model = CNNModel_BaseModelV3(50)
        #model = CNNModel_BaseModelV4(50)
        model = CNNModel_BaseModelV5(50)

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def initOptimizer(self):
        #return SGD(self.model.parameters(), lr=0.001, momentum=0.99)
        return Adam(self.model.parameters(), lr=3e-4, weight_decay=0.001)

    def initTransform(self):
        mean = [-8.8024]
        std = [20.8701]
        sample_rate = 44100

        data_transform = transforms.Compose([
            torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=128, n_fft=2048, hop_length=512, f_min=0,
                                                 f_max=8000).to(self.device),
            torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80).to(self.device),
            transforms.Normalize(mean, std).to(self.device)
        ])
        # original dimensions [1, 128, 431]
        crop_transform = transforms.Compose([
            transforms.RandomCrop(size=(100, 384)).to(self.device),
        ])

        base_data_transform = transforms.Compose([
            data_transform,
            crop_transform
        ])
        return base_data_transform

    def initAugmentedTransform(self, base_transform):

        data_transform = transforms.Compose([
            torchaudio.transforms.TimeMasking(time_mask_param=int(30 * (random.random() + 0.4))).to(self.device),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=int(10 * (random.random() + 0.4))).to(self.device),
            torchaudio.transforms.TimeMasking(time_mask_param=int(30 * (random.random() + 0.4))).to(self.device),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=int(10 * (random.random() + 0.4))).to(self.device)
        ])

        complete_transform = transforms.Compose([
            base_transform,
            data_transform,
        ])

        return complete_transform



    def initTrain(self, fold=None):
        assert 0 < fold <= 5, "Incorrect fold"
        batch_size = self.cli_args.batch_size

        dataset = ESC50Data_gpu(self.path, *self.columns, fold, False, self.transform, self.device)
        dataset_augmented = ESC50Data_gpu(self.path, *self.columns, fold, False, self.augmentedtransform,
                                          self.device)
        dataset_augmented1 = ESC50Data_gpu(self.path, *self.columns, fold, False, self.augmentedtransform,
                                           self.device)
        dataset_augmented2 = ESC50Data_gpu(self.path, *self.columns, fold, False, self.augmentedtransform,
                                           self.device)
        dataset_augmented3 = ESC50Data_gpu(self.path, *self.columns, fold, False, self.augmentedtransform,
                                           self.device)

        combined_dataset = ConcatDataset([dataset, dataset_augmented, dataset_augmented1, dataset_augmented2,
                                          dataset_augmented3])


        train_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=batch_size,
                                                   shuffle=False)
        return train_loader

    def initVal(self, fold=None):
        assert 0 < fold <= 5, "Incorrect fold"

        batch_size = self.cli_args.batch_size
        dataset_val = ESC50Data_gpu(self.path, *self.columns, fold, True, self.transform, self.device)

        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size,
                                                 shuffle=False)
        return val_loader

    def initTensorboardWriters(self, phase):
        log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)
        writer = None
        if phase == "train":
            writer = SummaryWriter(log_dir=log_dir + '-trn_cls-' + self.cli_args.comment)
        elif phase == "val":
            writer = SummaryWriter(log_dir=log_dir + '-val_cls-' + self.cli_args.comment)
        return writer


    def main(self):
        min_loss = 10.0
        for epoch in range(1, self.cli_args.epochs + 1):
            log.info(f"Time {datetime.datetime.now()}, Epoch {epoch}")
            # we get the mean of the batch metrics and the apply the mean of the 5 folds

            fold_metrics = torch.zeros(2, self.num_folds, 5, device=self.device)
            for fold in range(1, self.num_folds + 1):
                #log.info(f"Time {datetime.datetime.now()}, Fold {fold}/{self.num_folds}")
                train_loader = self.initTrain(fold)
                val_loader = self.initVal(fold)

                #print(len(train_loader.dataset), "  ", len(val_loader.dataset))
                for batch_idx, batch_tuple in enumerate(train_loader):
                    #print(batch_idx, "    ",datetime.datetime.now())

                    self.model.train()
                    loss = self.computeBatchLoss(batch_idx, batch_tuple)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                self.totalTrainingSamples_count += len(train_loader.dataset)

                self.model.eval()
                for name, loader in [("train", train_loader), ("val", val_loader)]:
                    batch_metrics = torch.zeros(len(loader), 5, device=self.device)  # we want to store 4 variables

                    with torch.no_grad():  # <1>
                        for batch_idx, batch_tuple in enumerate(loader):
                            self.computeBatchLoss(batch_idx, batch_tuple, batch_metrics, evaluation=True)

                    self.computeMetrics(name, batch_metrics, fold_metrics, fold-1)

            # evaluation of the epoch

            loss_val = self.logMetrics(epoch, fold_metrics, self.totalTrainingSamples_count)
            min_loss = min(min_loss, loss_val)

            self.saveModel(epoch, min_loss == loss_val)

            self.writer_val.close()
            self.writer_trn.close()

    def computeBatchLoss(self, batch_idx, batch_tuple, batch_metrics=None, evaluation=False):
        # returns the mean of the batch loss by default
        loss_func = nn.CrossEntropyLoss()

        imgs, labels = batch_tuple
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)
        outputs = self.model(imgs)
        loss = loss_func(outputs, labels)


        if evaluation:
            correct_predictions = (outputs.argmax(dim=1) == labels).sum().item()
            accuracy = correct_predictions / labels.shape[0]

            num_classes = 50
            true_positives = torch.zeros(num_classes)
            false_positives = torch.zeros(num_classes)
            false_negatives = torch.zeros(num_classes)

            for i in range(num_classes):
                true_positives[i] = ((outputs.argmax(dim=1) == i) & (labels == i)).sum().item()
                false_positives[i] = ((outputs.argmax(dim=1) == i) & (labels != i)).sum().item()
                false_negatives[i] = ((outputs.argmax(dim=1) != i) & (labels == i)).sum().item()

            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)

            for i in range(num_classes):
                if np.isnan(precision[i]):
                    precision[i] = 0.0
                if np.isnan(recall[i]):
                    recall[i] = 0.0

                #print(f"Class {i}: Precision: {precision[i]:.2f}, Recall: {recall[i]:.2f}")

            # Store metrics in the batch_metrics tensor, in our case it would be 320 *
            # compute precision and recall for all classes
            precision = float(precision.sum()) / precision.shape[0]
            recall = float(recall.sum()) / recall.shape[0]
            # f1 score
            if precision + recall != 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0

            batch_metrics[batch_idx, 0] = loss
            batch_metrics[batch_idx, 1] = accuracy
            batch_metrics[batch_idx, 2] = precision
            batch_metrics[batch_idx, 3] = recall
            batch_metrics[batch_idx, 4] = f1_score


        return loss


    def computeMetrics(self, phase, batch_metrics, fold_metrics, fold):
        #print(batch_metrics.shape, "    ", batch_metrics[:, 0])
        avg_loss = float(batch_metrics[:, 0].sum()) / batch_metrics.shape[0]
        avg_accuracy = float(batch_metrics[:, 1].sum()) / batch_metrics.shape[0]
        avg_precision = float(batch_metrics[:, 2].sum()) / batch_metrics.shape[0]
        avg_recall = float(batch_metrics[:, 3].sum()) / batch_metrics.shape[0]
        avg_f1_score = float(batch_metrics[:, 4].sum()) / batch_metrics.shape[0]

        if phase == 'val':
            fold_metrics[1, fold, 0] = avg_loss
            fold_metrics[1, fold, 1] = avg_accuracy
            fold_metrics[1, fold, 2] = avg_precision
            fold_metrics[1, fold, 3] = avg_recall
            fold_metrics[1, fold, 4] = avg_f1_score
        elif phase == 'train':
            fold_metrics[0, fold, 0] = avg_loss
            fold_metrics[0, fold, 1] = avg_accuracy
            fold_metrics[0, fold, 2] = avg_precision
            fold_metrics[0, fold, 3] = avg_recall
            fold_metrics[0, fold, 4] = avg_f1_score

    # we can try to write by batch instead of epoch
    def logMetrics(self, epoch, fold_metrics, totalTrainingSamples_count):
        for i in range(2):
            #print(fold_metrics[i, :, 0], "loss before dividing")
            avg_loss = float(fold_metrics[i, :, 0].sum()) / fold_metrics.shape[1]
            avg_accuracy = float(fold_metrics[i, :, 1].sum()) / fold_metrics.shape[1]
            avg_precision = float(fold_metrics[i, :, 2].sum()) / fold_metrics.shape[1]
            avg_recall = float(fold_metrics[i, :, 3].sum()) / fold_metrics.shape[1]
            avg_f1_score = float(fold_metrics[i, :, 4].sum()) / fold_metrics.shape[1]

            time = datetime.datetime.now()
            if i == 1:
                log.info(
                    "Time {} | Epoch {} | Phase {} | Loss: {:.4f} | Accuracy: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | F1 Score: {:.4f}".format(
                        time, epoch, "val", avg_loss, avg_accuracy, avg_precision, avg_recall, avg_f1_score
                    ))

                self.writer_val.add_scalar('Loss', avg_loss, totalTrainingSamples_count)
                self.writer_val.add_scalar('Accuracy', avg_accuracy, totalTrainingSamples_count)
                self.writer_val.add_scalar('Precision', avg_precision, totalTrainingSamples_count)
                self.writer_val.add_scalar('Recall', avg_recall, totalTrainingSamples_count)
                self.writer_val.add_scalar('F1 Score', avg_f1_score, totalTrainingSamples_count)

                loss_val = avg_loss
            else:
                log.info(
                    "Time {} | Epoch {} | Phase {} | Loss: {:.4f} | Accuracy: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | F1 Score: {:.4f}".format(
                        time, epoch, "train", avg_loss, avg_accuracy, avg_precision, avg_recall, avg_f1_score
                    ))

                self.writer_trn.add_scalar('Loss', avg_loss, totalTrainingSamples_count)
                self.writer_trn.add_scalar('Accuracy', avg_accuracy, totalTrainingSamples_count)
                self.writer_trn.add_scalar('Precision', avg_precision, totalTrainingSamples_count)
                self.writer_trn.add_scalar('Recall', avg_recall, totalTrainingSamples_count)
                self.writer_trn.add_scalar('F1 Score', avg_f1_score, totalTrainingSamples_count)


        return loss_val

    def saveModel(self, epoch, isBest=False):
        file_path = os.path.join(
            'models',
            self.cli_args.tb_prefix,
            '{}_{}.{}.state'.format(
                self.time_str,
                self.cli_args.comment,
                self.totalTrainingSamples_count,
            )
        )
        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)
        model = self.model

        state = {
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }
        torch.save(state, file_path)

        log.debug("Saved model params to {}".format(file_path))

        if isBest:
            best_path = os.path.join(
                'models',
                self.cli_args.tb_prefix,
                '{}_{}.{}.state'.format(
                    self.time_str,
                    self.cli_args.comment,
                    'best',
                )
            )
            shutil.copyfile(file_path, best_path)

            log.debug("Saved best model params to {}".format(best_path))


if __name__ == '__main__':
    TrainingApp().main()
