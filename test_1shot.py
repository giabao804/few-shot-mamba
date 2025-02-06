import torch
import numpy as np
import torch.nn as nn
import argparse
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
import function.function as function
import time
from tqdm import tqdm
import os
from function.function import ContrastiveLoss, seed_func, cal_metrics_fewshot
from CWRU.CWRU_dataset import CWRU
from HUST_bearing.HUST_dataset import HUSTbearing
from dataloader.dataloader import FewshotDataset
from torch.utils.data import DataLoader
from net.new_proposed import MainNet
from sklearn.metrics import confusion_matrix
import argparse
import torch.nn as nn
import numpy as np
import librosa
import cv2
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from IPython.display import clear_output





parser = argparse.ArgumentParser(description='Bearing Faults Project Configuration')
parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
parser.add_argument('--h', type=int, default=16, help='Height of the input image')
parser.add_argument('--w', type=int, default=16, help='Width of the input image')
parser.add_argument('--c', type=int, default=64, help='Number of channels of the input image')
parser.add_argument('--dataset', choices=['HUST_bearing', 'CWRU', 'PDB'], help='Choose dataset for training')
parser.add_argument('--training_samples_CWRU', type=int, default=30, help='Number of training samples for CWRU')
parser.add_argument('--training_samples_PDB', type=int, default=195, help='Number of training samples for PDB')
parser.add_argument('--training_samples_HUST', type=int, default=168, help='Number of training samples for HUST_bearing')
parser.add_argument('--model_name', type=str, help='Model name')
parser.add_argument('--episode_num_train', type=int, default=130, help='Number of training episodes')
parser.add_argument('--episode_num_test', type=int, default=150, help='Number of testing episodes')
parser.add_argument('--way_num_CWRU', type=int, default=10, help='Number of classes for CWRU')
parser.add_argument('--noise_DB', type=str, default=None, help='Noise database')
parser.add_argument('--way_num_PDB', type=int, default=13, help='Number of classes for PDB')
parser.add_argument('--spectrum', action='store_true', help='Use spectrum')
parser.add_argument('--way_num_HUST', type=int, default=7, help='Number of classes for HUST')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda or cpu)')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--path_weights', type=str, default='checkpoints/', help='Path to weights')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--step_size', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--loss1', default=ContrastiveLoss())
parser.add_argument('--loss2', default=nn.CrossEntropyLoss())
parser.add_argument('--data_path', default="/content/drive/MyDrive/Bearing_Faults_CovaMNET/HUST bearing dataset/", help="data path")
parser.add_argument('--cfs_matrix', action='store_false', help="Print confusion matrix")
parser.add_argument('--train_mode', action='store_false', help="Select train mode")
args = parser.parse_args()

print(args)
#---------------------------------------------------Load dataset-----------------------------------------------------------------------------------------:
if args.dataset == 'CWRU':
    window_size = 2048
    split = args.training_samples_CWRU//30
    data = CWRU(split, ['12DriveEndFault'], ['1772', '1750', '1730'], window_size)
    data.X_train = data.X_train.astype(np.float32)
    data.X_test = data.X_test.astype(np.float32)
    train_data_CWRU = torch.from_numpy(data.X_train)
    train_label_CWRU = torch.from_numpy(data.y_train)
    test_data_CWRU = torch.from_numpy(data.X_test)
    test_label_CWRU = torch.from_numpy(data.y_test)
    train_data_CWRU = train_data_CWRU.reshape([args.training_samples_CWRU,4096])
    test_data_CWRU = test_data_CWRU.reshape([750,4096])

    if args.noise_DB != None:
        snr_dB = args.noise_DB
        data.add_noise_to_test_data(snr_dB, 0.001)
        noisy_test_data = data.X_test_noisy.reshape([750,4096])

        if args.spectrum != None:
            train_data_CWRU = function.to_spectrum(train_data_CWRU)
            test_data_CWRU = function.to_spectrum(noisy_test_data)
        else:
            train_data_CWRU = train_data_CWRU.reshape(train_data_CWRU.shape[0], 1, 64, 64)
            test_data_CWRU = train_data_CWRU.reshape(test_data_CWRU.shape[0], 1, 64, 64)

    else:
        if args.spectrum != None:
            train_data_CWRU = function.to_spectrum(train_data_CWRU)
            test_data_CWRU = function.to_spectrum(test_data_CWRU)
        else:
            train_data_CWRU = train_data_CWRU.reshape(train_data_CWRU.shape[0], 1, 64, 64)
            test_data_CWRU = test_data_CWRU.reshape(test_data_CWRU.shape[0], 1, 64, 64)


    train_dataset_CWRU = FewshotDataset(train_data_CWRU, train_label_CWRU, episode_num=args.episode_num_train, way_num=args.way_num_CWRU, shot_num=1, query_num=1)
    train_dataloader_CWRU = DataLoader(train_dataset_CWRU, batch_size=args.batch_size, shuffle=True)
    test_dataset_CWRU = FewshotDataset(test_data_CWRU, test_label_CWRU, episode_num=args.episode_num_test, way_num=args.way_num_CWRU, shot_num=1, query_num=1)
    test_dataloader_CWRU = DataLoader(test_dataset_CWRU, batch_size=args.batch_size, shuffle=False)
    clear_output()
    # Testing phase
    print('Load_model_from_checkpoint.....')
    seed_func()
    net = MainNet().to(args.device)
    net = torch.load(args.best_weight)
    print('Loading successfully!')
    acc, f1, recall = cal_metrics_fewshot(test_dataloader_CWRU, net,args.device, 10)
    print(f'Accuracy on the test set: {acc:.4f}')
    print(f'F1_score on the test set: {f1:.4f}')
    print(f'Recall on the test set: {recall:.4f}')



if args.dataset == 'HUST_bearing':
    def to_spectrum(data, h=64, w=64, sigma=0.6):
        spectrograms = []

        for i in range(data.shape[0]):
            signal = data[i, :]
            signal = np.array(signal)
            spectrogram = librosa.stft(signal, n_fft=512, hop_length=512)
            spectrogram = np.abs(spectrogram) ** 2
            log_spectrogram = librosa.power_to_db(spectrogram)
            log_spectrogram = cv2.resize(log_spectrogram, (h, w))
            smoothed_spectrogram = gaussian_filter(log_spectrogram, sigma=sigma)
            spectrograms.append(smoothed_spectrogram)

        data = np.stack(spectrograms).astype(np.float32)
        data = torch.from_numpy(data).unsqueeze(dim=1)

        return data
    HUST_data = HUSTbearing(data_dir=args.data_path)
    print(args.data_path)
    print('Loading data................!!')
    HUST_data.load_data()
    data = HUST_data.x_train.reshape(HUST_data.x_train.shape[0], -1)
    label = HUST_data.y_train
    data = to_spectrum(data)
    print(data.shape)
    print(label.shape)
    if args.training_samples_HUST == 16800:
    
        train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.1, random_state=42, stratify=label)

        test_size = args.training_samples_HUST/train_data.shape[0]
        _, x_train_limited,_, y_train_limited = train_test_split(train_data, train_label, test_size=test_size, random_state=42, stratify=train_label)
        test_data, _, test_label, _ = train_test_split(test_data, test_label, test_size=2/3, random_state=42, stratify=test_label) 
    else:
        train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2, random_state=42, stratify=label)

        test_size = args.training_samples_HUST/train_data.shape[0]
        _, x_train_limited,_, y_train_limited = train_test_split(train_data, train_label, test_size=test_size, random_state=42, stratify=train_label)
        test_data, _, test_label, _ = train_test_split(test_data, test_label, test_size=5/6, random_state=42, stratify=test_label) 

    
    print('Data train for limited case shape:', x_train_limited.shape)
    print('Label train for limited case shape', y_train_limited.shape)
    print('Data test for limited case shape:', test_data.shape)
    print('Label test for limited case shape', test_label.shape)

    plt.imshow(x_train_limited[0].squeeze(0), cmap='gray')

    

    train_dataset_HUST = FewshotDataset(x_train_limited, y_train_limited, episode_num=args.episode_num_train, way_num=args.way_num_HUST, shot_num=args.shot_num, query_num=1)
    train_dataloader_HUST = DataLoader(train_dataset_HUST, batch_size=args.batch_size, shuffle=True)
    test_dataset_HUST = FewshotDataset(test_data, test_label, episode_num=args.episode_num_test, way_num=args.way_num_HUST, shot_num=args.shot_num, query_num=1)
    test_dataloader_HUST = DataLoader(test_dataset_HUST, batch_size=args.batch_size, shuffle=False)   

    net = MainNet().to(args.device)
    net = torch.load(args.best_weight)
    acc, f1, recall = cal_metrics_fewshot(test_dataloader_HUST, net,args.device, 7)

    print(f'Accuracy on the test set: {acc:.4f}')
    print(f'F1_score on the test set: {f1:.4f}')
    print(f'Recall on the test set: {recall:.4f}')



