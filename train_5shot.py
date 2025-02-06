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
from function.function import  cal_accuracy_fewshot_5shot, predicted_fewshot, ContrastiveLoss, seed_func, cal_accuracy_fewshot, convert_for_5shots, predicted_fewshot_5shot, cal_metrics_5shot
from CWRU.CWRU_dataset import CWRU
import os
from HUST_bearing.HUST_dataset import HUSTbearing
from dataloader.dataloader import FewshotDataset
from torch.utils.data import DataLoader
from net.new_proposed import MainNet, Baseline

from sklearn.metrics import confusion_matrix
import argparse
import torch.nn as nn
import numpy as np
import librosa
import cv2
import torch
import numpy as np
import scipy.io
import scipy.io as sio
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


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
parser.add_argument('--noise_DB', type=float, default=None, help='Noise database')
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
parser.add_argument('--shot_num', type=int, default=5, help='Number of samples per class')
args = parser.parse_args()

print(args)
#---------------------------------------------------Load dataset-----------------------------------------------------------------------------------------:
if args.dataset == 'CWRU':
    window_size = 2048
    split = args.training_samples_CWRU//30
    data = CWRU(split, ['12DriveEndFault'], ['1772', '1750', '1730'], window_size)
    data.nclasses,data.classes,len(data.X_train),len(data.X_test)
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

    print('Shape of CWRU train data:',train_data_CWRU.shape)
    print('Shape of CWRU test data:',test_data_CWRU.shape)
    print('End Loading CWRU')
    
    train_dataset_CWRU = FewshotDataset(train_data_CWRU, train_label_CWRU, episode_num=args.episode_num_train, way_num=args.way_num_CWRU, shot_num=args.shot_num, query_num=1)
    train_dataloader_CWRU = DataLoader(train_dataset_CWRU, batch_size=args.batch_size, shuffle=True)
    test_dataset_CWRU = FewshotDataset(test_data_CWRU, test_label_CWRU, episode_num=args.episode_num_test, way_num=args.way_num_CWRU, shot_num=args.shot_num, query_num=1)
    test_dataloader_CWRU = DataLoader(test_dataset_CWRU, batch_size=args.batch_size, shuffle=False)

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


if args.dataset == 'PDB':
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
    print('Loading data................!!')
    data = torch.load(f'{args.data_path}data.pt')
    label = torch.load(f'{args.data_path}label.pt')
    data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    data = to_spectrum(data)
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42, stratify=label)
    test_size = args.training_samples_PDB/x_train.shape[0]
    _, x_train_limited,_, y_train_limited = train_test_split(x_train, y_train, test_size=test_size, random_state=42, stratify=y_train)
    print('Data train for limited case shape:', x_train_limited.shape)
    print('Label train for limited case shape', y_train_limited.shape)
    train_dataset_PDB = FewshotDataset(x_train_limited, y_train_limited, episode_num=args.episode_num_train, way_num=args.way_num_PDB, shot_num=args.shot_num, query_num=1)
    train_dataloader_PDB = DataLoader(train_dataset_PDB, batch_size=args.batch_size, shuffle=True)
    test_dataset_PDB = FewshotDataset(x_test, y_test, episode_num=args.episode_num_test, way_num=args.way_num_PDB, shot_num=args.shot_num, query_num=1)
    test_dataloader_PDB = DataLoader(test_dataset_PDB, batch_size=args.batch_size, shuffle=False)


def train_and_test_model(net,
                         train_dataloader,
                         test_loader,
                         training_samples,
                         num_epochs = args.num_epochs,
                         lr = args.lr,
                         loss1 = args.loss1,
                         path_weight = args.path_weights,
                         num_samples = args.training_samples_HUST,
                         num_classes = args.way_num_HUST):
    device = args.device
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    loss1.to(device)
    full_loss = []
    full_metric = {'full_acc' :[], 'full_f1': [], 'full_recall': []}
    pred_metric = {'pred_acc': 0, 'pred_f1': 0, 'pred_recall': 0}    

    cumulative_time = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        running_loss = 0
        num_batches = 0
        optimizer.zero_grad()
        print('='*50, 'Epoch:', epoch, '='*50)
        with tqdm(train_dataloader, desc=f'Epoch {epoch}/{num_epochs}', unit='batch') as t:
            for query_images, query_targets, support_images, support_targets in t:
                q = query_images.permute(1, 0, 2, 3, 4).to(device)
                s = convert_for_5shots(support_images, support_targets, device)
                targets = query_targets.to(device)
                targets = targets.permute(1, 0)
                for i in range(len(q)):
                    out, _, _ = net(q[i], s)
                    target = targets[i].long()
                    loss = loss1(out, target)
                    loss.backward()
                    running_loss += loss.detach().item()
                    num_batches += 1
                optimizer.step()
                optimizer.zero_grad()
                t.set_postfix(loss=running_loss / num_batches)

        elapsed_time = time.time() - start_time
        cumulative_time += elapsed_time
        cumulative_minutes = cumulative_time / 60
        print(f"Epoch {epoch}/{num_epochs} completed in {cumulative_minutes:.2f} minutes")

        scheduler.step()

        with torch.no_grad():
            total_loss = running_loss / num_batches
            full_loss.append(total_loss)
            print('------------Testing on the test set-------------')
            acc, f1, recall = cal_metrics_5shot(test_loader, net, device, num_classes)
            full_metric['full_acc'].append(acc)
            full_metric['full_f1'].append(f1)
            full_metric['full_recall'].append(recall)
            print(f'Accuracy on the test set: {acc:.4f}')
            print(f'F1_score on the test set: {f1:.4f}')
            print(f'Recall on the test set: {recall:.4f}')
            if acc > pred_metric['pred_acc']:
                if epoch >= 2:
                    os.remove(path_weight + model_name)
                pred_metric['pred_acc'] = acc
                pred_metric['pred_f1'] = f1
                pred_metric['pred_recall'] = recall
                model_name = f'{args.model_name}_5shot_recall_{recall:.4f}_{training_samples}samples.pth'
                torch.save(net, path_weight + model_name)
                print(f'=> Save the best model with accuracy: {acc:.4f}')
        torch.cuda.empty_cache()

    return full_loss, full_metric, model_name, pred_metric['pred_acc'], pred_metric['pred_f1'], pred_metric['pred_recall']

#----------------------------------------------------Training phase--------------------------------------------------#
seed_func()
print("train or val:")
if args.train_mode:
  net = MainNet()
  net = net.to(args.device)
  print('training in case of 5 shot.........................!!')
  if args.dataset == 'CWRU':
    _,_,model_name, acc, vec_q, vec_s =  train_and_test_model(net,
                        train_dataloader = train_dataloader_CWRU,
                        test_loader = test_dataloader_CWRU,
                        training_samples = args.training_samples_CWRU,
                        num_epochs = args.num_epochs,
                        lr = args.lr,
                        loss1 = args.loss1,
                        path_weight = args.path_weights,
                        num_samples = args.training_samples_CWRU,
                        num_classes = args.way_num_CWRU)
  elif args.dataset == 'PDB':
    _,_,model_name, acc, vec_q, vec_s =  train_and_test_model(net,
                        train_dataloader = train_dataloader_PDB,
                        test_loader = test_dataloader_PDB,
                        training_samples = args.training_samples_PDB,
                        num_epochs = args.num_epochs,
                        lr = args.lr,
                        loss1 = args.loss1,
                        path_weight = args.path_weights,
                        num_samples = args.training_samples_PDB,
                        num_classes = args.way_num_PDB)
    
  elif args.dataset == 'HUST_bearing':
    print("Training with HUST bearing dataset in case of 5 shot....")
    _,_,model_name, acc, vec_q, vec_s = train_and_test_model(net,
                        train_dataloader = train_dataloader_HUST,
                        test_loader = test_dataloader_HUST,
                        training_samples = args.training_samples_HUST,
                        num_epochs = args.num_epochs,
                        lr = args.lr,
                        loss1 = args.loss1,
                        path_weight = args.path_weights,
                        num_samples = args.training_samples_HUST,
                        num_classes = args.way_num_HUST)    

  print('end training...................!!')

if args.cfs_matrix:
    print("validating...")
    faults_idx = {
    'Normal': 0,
    '0.007-Ball': 1,
    '0.014-Ball': 2,
    '0.021-Ball': 3,
    '0.007-Inner': 4,
    '0.014-Inner': 5,
    '0.021-Inner': 6,
    '0.007-Outer': 7,
    '0.014-Outer': 8,
    '0.021-Outer': 9,
#     '0.007-OuterRace3': 10,
#     '0.014-OuterRace3': 11,
#     '0.021-OuterRace3': 12,
#     '0.007-OuterRace12': 13,
#     '0.014-OuterRace12': 14,
#     '0.021-OuterRace12': 15,
}


    net = MainNet()
  
    saved_weights_path = f"{args.path_weights}{model_name}"
    net = torch.load(saved_weights_path)
    if args.dataset == 'CWRU':
        true_labels, predicted, vec_q, vec_s = predicted_fewshot_5shot(test_dataloader_CWRU, net, args.device)
    elif args.dataset == 'HUST':   
        true_labels, predicted, vec_q, vec_s = predicted_fewshot_5shot(test_dataloader_HUST, net, args.device)

    faults_labels = {v: k for k, v in faults_idx.items()}
    unique_labels = np.unique(true_labels)
    tick_labels = [faults_labels[label] for label in unique_labels]  
    print(tick_labels) 

    predicted = predicted.squeeze()
    predicted_labels = np.argmax(predicted, axis = 1)
    confusion = confusion_matrix(true_labels.squeeze(), predicted_labels)
    plt.figure(figsize=(10, 8) )
    plt.imshow(confusion, cmap='RdPu')
    plt.colorbar()
    plt.xlabel('Predicted Labels', fontsize=16)
    plt.ylabel('Actual Labels', fontsize=16)
    plt.title('Confusion Matrix', fontsize=16)

    if args.dataset == 'CWRU':
        total = np.sum(confusion)/args.way_num_CWRU
    elif args.dataset == 'HUST_bearing':
        total = np.sum(confusion)/args.way_num_HUST    

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            count = confusion[i, j]
            percent = (count / total) * 100
            text_color = 'white' if count > 50 else 'black'
            # if i != j:
            #   # plt.text(j, i, str(confusion[i, j]), ha='center', va='center', color=text_color, fontsize=14)
            #   plt.text(j, i, f'{count}\n({percent:.1f})%', ha='center', va='center', color=text_color)

            # else:
            plt.text(j, i - 0.1, f'{count}', ha='center', va='center', color=text_color, fontsize=11)
            plt.text(j, i + 0.2, f'({percent:.1f}%)', ha='center', va='center', color=text_color, fontsize=9)  # fontsize for count



    tick_locations = np.arange(len(unique_labels))
    plt.xticks(tick_locations, tick_labels, rotation=45, ha='right', fontsize=9)
    plt.yticks(tick_locations, tick_labels, rotation=45, ha='right', fontsize=9)

  # Save the figure
    if args.dataset == 'CWRU':
        save_path = f"{args.path_weights}cfs_{args.training_samples_CWRU}_{acc}.png"
    elif args.dataset == 'HUST_bearing':
        save_path = f"{args.path_weights}cfs_{args.training_samples_HUST}_{acc}.png"
  
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(predicted)
    plt.figure(figsize=(5, 5))
    plt.grid(True, ls='--', alpha=0.5)
    unique_labels = np.unique(true_labels)
    num_classes = len(unique_labels)
    color_map = plt.cm.get_cmap('Paired', num_classes)
    for i, label in enumerate(unique_labels):
        class_indices = np.where(true_labels == label)
        plt.scatter(tsne_results[class_indices, 0], tsne_results[class_indices, 1],
                label=f'Class {label}', color=color_map(i), s=30, alpha=0.8, linewidths=2)
    plt.tight_layout()

    if args.dataset == 'CWRU':
        save_path = f"{args.path_weights}tsne_{args.training_samples_CWRU}_{acc}.png"
    elif args.dataset == 'HUST_bearing':
        save_path = f"{args.path_weights}tsne_{args.training_samples_HUST}_{acc}.png"

    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

