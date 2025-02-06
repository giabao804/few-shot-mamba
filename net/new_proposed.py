import torch
import torch.nn as nn
import torch.nn.functional as F
import functools 
import numpy as np
import torchmetrics
from torchmetrics.regression import MinkowskiDistance
from net.mamba import SS_Conv_SSM
from net.feature_extractor import Feature_Extractor, ConvMixer, SB
from net.glca import ChannelAttention
# from net.corr_sim import CorrSim, CAM
 
class CovaBlock(nn.Module):
    def __init__(self):
        super(CovaBlock, self).__init__()

    def cal_covariance(self, input):
        CovaMatrix_list = []
        for i in range(len(input)):
            support_set_sam = input[i]
            B, C, h, w = support_set_sam.size()

            support_set_sam = support_set_sam.permute(1, 0, 2, 3)
            support_set_sam = support_set_sam.contiguous().view(C, -1)
            mean_support = torch.mean(support_set_sam, 1, True)
            support_set_sam = support_set_sam - mean_support

            covariance_matrix = support_set_sam @ torch.transpose(support_set_sam, 0, 1)
            covariance_matrix = torch.div(covariance_matrix, h * w * B - 1)
            CovaMatrix_list.append(covariance_matrix)
        return CovaMatrix_list

    def cal_similarity(self, input, CovaMatrix_list):
        B, C, h, w = input.size()
        Cova_Sim = []

        for i in range(B):
            query_sam = input[i]
            query_sam = query_sam.view(C, -1)
            query_sam_norm = torch.norm(query_sam, 2, 1, True)
            query_sam = query_sam / query_sam_norm

            if torch.cuda.is_available():
                mea_sim = torch.zeros(1, len(CovaMatrix_list) * h * w).cuda()
            else:
                mea_sim = torch.zeros(1, len(CovaMatrix_list) * h * w)
            for j in range(len(CovaMatrix_list)):
                temp_dis = torch.transpose(query_sam, 0, 1) @ CovaMatrix_list[j] @ query_sam
                mea_sim[0, j * h * w:(j + 1) * h * w] = temp_dis.diag()

            Cova_Sim.append(mea_sim.view(1, -1))

        Cova_Sim = torch.cat(Cova_Sim, 0)

        return Cova_Sim

    def forward(self, x1, x2):
        CovaMatrix_list = self.cal_covariance(x2)
        Cova_Sim = self.cal_similarity(x1, CovaMatrix_list)

        return Cova_Sim
    
import functools

class MainNet(nn.Module):
    def __init__(self, h=16, w =16, c=64, dim=64, norm_layer=nn.BatchNorm2d, num_classes=13, alpha1=0.8, alpha2 =0.2):
        super(MainNet, self).__init__()
        self.h = h
        self.w = w
        self.c = c

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d


        # self.features = EF()
        self.features1 = ConvMixer(patch_size=4)
        self.features2 = Feature_Extractor()
        self.lower = ChannelAttention(dim//4,3)
        self.upper = SS_Conv_SSM(hidden_dim=dim, d_state=16)
        self.covariance = CovaBlock()
        self.SB = SB(64)

        self.classifier1 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
            nn.Conv1d(1, 1, kernel_size=self.h*self.w, stride=self.h*self.w, bias=use_bias),
        )
        # self.sim = CAM()
        # self.alpha1 = alpha1
        # self.alpha2 = alpha2

    def forward(self, input1, input2):
      

        f1 = self.upper(self.features1(input1))
        f2 = self.lower(self.features2(input1))
        S = []

        q, vec_q = self.SB(f1,f2)
        # q= f1+f2
      
        for i in range(len(input2)):

            s1 = self.upper(self.features1(input2[i]))
            s2 = self.lower(self.features2(input2[i]))
            s, vec_s= self.SB(s1,s2)
            S.append(s)
        #   s = s1+s2
   
        x1 = self.covariance(q, S)
        
        # x2 = self.sim(q.unsqueeze(dim=0), torch.cat(S,0))

        x1 = self.classifier1(x1.view(x1.size(0), 1, -1))
        output = x1.squeeze(1)


        return output, vec_q, vec_s


import torch
import torch.nn as nn
import functools

class Baseline(nn.Module):
    def __init__(self, h: int = 16, w: int = 16, c: int = 64, dim: int = 64, 
                 norm_layer: nn.Module = nn.BatchNorm2d, num_classes: int = 10, model_state: int = 1):
        super(Baseline, self).__init__()
        self.h = h
        self.w = w
        self.c = c
        self.dim = dim

        # Determine if bias is needed based on norm layer
        use_bias = (norm_layer == nn.InstanceNorm2d or 
                    (isinstance(norm_layer, functools.partial) and norm_layer.func == nn.InstanceNorm2d))

        # Feature extraction layers
        self.features1 = ConvMixer(patch_size=4)
        self.features2 = Feature_Extractor()
        self.covariance = CovaBlock()

        # Attention and feature blocks
        self.SB = SB(64)
        self.upper = SS_Conv_SSM(hidden_dim=dim, d_state=16)
        self.lower = ChannelAttention(dim // 4, 3)

        # Classifier layer
        self.classifier1 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(),
            nn.Conv1d(1, 1, kernel_size=h * w, stride=h * w, bias=use_bias)
        )

        # Model state
        self.model_state = model_state

    def extract_features(self, input1, input2, use_upper=False, use_lower=False):
        """ Helper function to extract and combine features """
        f1 = self.upper(self.features1(input1)) if use_upper else self.features1(input1)
        f2 = self.lower(self.features2(input1)) if use_lower else self.features2(input1)
        q = f1 + f2
        S = []

        for inp in input2:
            s1 = self.upper(self.features1(inp)) if use_upper else self.features1(inp)
            s2 = self.lower(self.features2(inp)) if use_lower else self.features2(inp)
            S.append(s1 + s2)

        return q, S

    def forward(self, input1: torch.Tensor, input2: list) -> torch.Tensor:
        if self.model_state == 1:
            q, S = self.extract_features(input1, input2)
        elif self.model_state == 2:
            q, S = self.extract_features(input1, input2, use_upper=True)
        elif self.model_state == 3:
            q, S = self.extract_features(input1, input2, use_lower=True)
        elif self.model_state == 4:
            f1, f2 = self.features1(input1), self.features2(input1)
            q, vec_q = self.SB(f1, f2)
            S = [self.SB(self.features1(inp), self.lower(self.features2(inp)))[0] for inp in input2]
        elif self.model_state == 5:
            q, S = self.extract_features(input1, input2, use_upper=True, use_lower=True)
        elif self.model_state == 6:
            f1 = self.upper(self.features1(input1))
            f2 = self.features2(input1)
            q, vec_q = self.SB(f1, f2)
            S = [self.SB(self.upper(self.features1(inp)), self.features2(inp))[0] for inp in input2]
        elif self.model_state == 7:
            f1 = self.features1(input1)
            f2 = self.lower(self.features2(input1))
            q, vec_q = self.SB(f1, f2)
            S = [self.SB(self.features1(inp), self.lower(self.features2(inp)))[0] for inp in input2]
        else:
            raise ValueError(f"Unsupported model_state: {self.model_state}")

        x1 = self.covariance(q, S)
        x1 = self.classifier1(x1.view(x1.size(0), 1, -1))
        output = x1.squeeze(1)

        return output, output, output
