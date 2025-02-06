import numpy as np
import scipy.io
import scipy.io as sio
import os,re
import errno
import urllib.request as urllib
from scipy.io import loadmat
from sklearn.utils import shuffle
import random
import torch
import librosa
from IPython.display import clear_output
import argparse
# from train_1shot import args



import re
def Get_Label(file_name):
    fault = re.search(r'([A-Za-z]+)', file_name).group()
    return fault
faults_map = {
    'N': 0,
    'I': 1,
    'O': 2,
    'B': 3,
    'IO': 4,
    'IB': 5,
    'OB': 6,
}

class HUSTbearing:
        def __init__(self, data_dir, segment_length=2048, overlap=0.75):
            self.segment_length = segment_length
            self.overlap = overlap
            self.data_dir = data_dir
            self.x_train = torch.rand(21000, 2048, 2)
            self.y_train = torch.zeros(21000)

        def load_data(self):
            file_list = os.listdir(self.data_dir)
            mat_files = [file for file in file_list if file.endswith('.mat')]
            idx = 0
            count_file = 0

            for file_name in mat_files:
                file_path = os.path.join(self.data_dir, file_name)
                data = scipy.io.loadmat(file_path)
                bearing_data = data['data']
                bearing_data = torch.from_numpy(bearing_data)
            #y_train[idx] = faults_map[Get_Label(file_name)]
                segments = []
                total_length = bearing_data.shape[0]
                stride = int(self.segment_length * (1 - self.overlap))
                check_label = faults_map[Get_Label(file_name)]
                if check_label == 3:
                    num_segments = 250
                elif check_label == 5:
                    num_segments = 250
                else:
                    num_segments = 200
                for i in range(num_segments):
                    self.y_train[idx] = faults_map[Get_Label(file_name)]
                    start = i * stride
                    end = start + self.segment_length
                    segment = bearing_data[start:end]
                    segment = torch.cat((segment, segment), dim=1)
                    self.x_train[idx] = segment
                    idx += 1

        


