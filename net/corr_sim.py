import torch
import torch.nn as nn
import torch.nn.functional as F
import functools 

import torch
import torch
import torch.nn as nn
import argparse

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
# parser.add_argument('--loss1', default=ContrastiveLoss())
parser.add_argument('--loss2', default=nn.CrossEntropyLoss())
parser.add_argument('--data_path', default="/content/drive/MyDrive/Bearing_Faults_CovaMNET/HUST bearing dataset/", help="data path")
parser.add_argument('--cfs_matrix', action='store_false', help="Print confusion matrix")
parser.add_argument('--train_mode', action='store_false', help="Select train mode")
args = parser.parse_args()

import torch.nn.functional as F

class CosineSimilarity(nn.Module):
    def __init__(self):
        super(CosineSimilarity, self).__init__()

    def forward(self, a, b):
        
        a = torch.tensor(a, requires_grad=True, dtype=torch.float32)
        b = torch.tensor(b, requires_grad=True, dtype=torch.float32)
        
        
        cosine_sim = F.cosine_similarity(a, b, dim=1)
        
        return cosine_sim

class CorrSim(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
      

    def cal_similarity(self, input, vec_support_list):
       
        B, C, h, w = input.size()
        vec_similar = []

        
        for i in range(B):
            query_sam = input[i]
            query_sam = query_sam.view(C, -1)
            query_sam_norm = torch.norm(query_sam, 2, 1, True)
            query_sam = query_sam / query_sam_norm
            query_sam = query_sam.sum(dim=0, keepdim=True)

            if torch.cuda.is_available():
                vec_sim = torch.zeros(1, len(vec_support_list)*C).cuda()
            else:
                vec_sim = torch.zeros(1, len(vec_support_list)*C)
            
            for j in range(len(vec_support_list)):
                s = vec_support_list[j]
                s = s.squeeze(dim=0).sum(dim=0, keepdim=True)
                # similar = corrcoef(query_sam, s)
                if torch.cuda.is_available():
                    similar = F.cosine_similarity(query_sam.cuda(), s.cuda())
                else:
                    similar = F.cosine_similarity(query_sam, s)  

                vec_sim[0,j*C:(j+1)*C] = similar
            
            vec_similar.append(vec_sim)

        vec_similar = torch.cat(vec_similar, 0)    

        return vec_similar

    def cal_vec(self, input):
        vector_list = []
        for i in range(len(input)):
            support_set_sam = input[i]
            B, C, h, w = support_set_sam.size()

            support_set_sam = support_set_sam.permute(1, 0, 2, 3)
            support_set_sam = support_set_sam.contiguous().view(B, C, -1)
            mean_support = torch.mean(support_set_sam, 1, True)
            support_set_sam = support_set_sam - mean_support

            if torch.cuda.is_available():
                vector_list.append(support_set_sam.cuda())
            else:
                vector_list.append(support_set_sam)

            # vector_list.append(support_set_sam.cuda())
        return vector_list


    def forward(self, x1, x2):
        vec_list = self.cal_vec(x2)
        mea_dis = self.cal_similarity(x1, vec_list)
        # mea_sim = self.softmax(mea_dis)
        # mea_sim = 1/(1+mea_dis)
        return mea_dis
        

import math

class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization.

    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
    - in_c (int): number of input channels.
    - out_c (int): number of output channels.
    - k (int or tuple): kernel size.
    - s (int or tuple): stride.
    - p (int or tuple): padding.
    """
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class CAM(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(CAM, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.conv1 = ConvBlock(256, 16, 1)
        self.conv2 = nn.Conv2d(16, 256, 1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        self.classifier = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
            nn.Conv1d(1, 1, kernel_size=args.h*args.w, stride=args.h*args.w, bias=use_bias),
        )


    def get_attention(self, a):

        input_a = a

        a = a.mean(3)
        a = a.transpose(1, 3)
        a = F.relu(self.conv1(a))
        a = self.conv2(a)
        a = a.transpose(1, 3)
        a = a.unsqueeze(3)

        a = torch.mean(input_a * a, -1)
        a = F.softmax(a / 0.025, dim=-1) + 1
        return a

    def forward(self, f1, f2):

        n1, b, c, h, w = f1.size()

        n2 = f2.size(0)

        f1 = f1.view(b, n1, c, -1)
        f2 = f2.view(b, n2, c, -1)


        f1_norm = F.normalize(f1, p=2, dim=2, eps=1e-12)
        f2_norm = F.normalize(f2, p=2, dim=2, eps=1e-12)

        f1_norm = f1_norm.transpose(2, 3).unsqueeze(2)
        f2_norm = f2_norm.unsqueeze(1)

        a1 = torch.matmul(f1_norm, f2_norm)

        # a2 = a1.transpose(3, 4)
        a1 = self.get_attention(a1)
        # a2 = self.get_attention(a2)
        a1 = self.classifier(a1.squeeze(dim=0).view(1,-1))
        
        return a1

    
