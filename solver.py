import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from model.RevIN import RevIN
import time
from utils.utils import *
from model.PPLAD import PPLAD
from data_factory.data_loader import get_loader_segment
from einops import rearrange
from metrics.metrics import *
import warnings

warnings.filterwarnings('ignore')



def norm(x):
    mean = torch.mean(x, dim=-1, keepdim=True)
    stddev = torch.std(x, dim=-1, keepdim=True)

    # 对最后一个维度进行归一化，同时添加一个小的正则项以避免除以零
    normalized_tensor = (x - mean) / (stddev + 1e-5)
    return  normalized_tensor
def minmax_norm(x):
    min, _= torch.min(x,dim=-1,keepdim=True)
    max, _ = torch.max(x,dim=-1,keepdim=True)
    return (x - min) / (max-min+1e-5)
def my_kl_loss(p, q):  # 128 1 100 100
    Min= torch.min(p)
    Min_1= torch.min(q)
    Min = torch.min(Min,Min_1)
    offset = max(-Min.item(), 0)
    res = p * (torch.log(p + 0.001+offset) - torch.log(q + 0.001+offset))
    return torch.sum(res, dim=2)  # 128 1 100->128 100


def my_kl_loss_1(p, q):  # 128 1 100 100
    res = p * (torch.log(p + 0.001) - torch.log(q + 0.001))
    return torch.mean(torch.sum(res, dim=1), dim=-1)  # 128 1 100->128 100

def normalize_vector(x):
    B,L,N = x.shape
    Sum = torch.sum(x, dim=-1).unsqueeze(-1).repeat(1,1,N)
    return x/Sum
def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def cal_similar(similar,in_size,x,num):
    if similar == "MSE":
        criterion_keep = nn.MSELoss(reduction='none')
        in_size = torch.sum(criterion_keep(in_size, x.unsqueeze(-1).repeat(1, 1, 1, num)), dim=2)
    elif similar == "cos":
        in_size = F.cosine_similarity(in_size, x.unsqueeze(-1).repeat(1, 1, 1, num), dim=2, eps=1e-8)
    elif similar == "kl":
        in_size = my_kl_loss(in_size, x.unsqueeze(-1).repeat(1, 1, 1, num))
    return in_size
class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                               win_size=self.win_size, mode='train', dataset=self.dataset, )
        self.vali_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                              win_size=self.win_size, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                              win_size=self.win_size, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                              win_size=self.win_size, mode='thre', dataset=self.dataset)
        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
            self.criterion_keep = nn.L1Loss(reduction='none')
        elif self.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()
            self.criterion_keep= nn.MSELoss(reduction='none')
    def build_model(self):
        self.model = PPLAD(batch_size=self.batch_size,win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c,
                                d_model=self.d_model, local_size=self.local_size,global_size=self.global_size,
                                channel=self.input_c)

        if torch.cuda.is_available():
            self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        op="train"
        time_now = time.time()
        train_steps = len(self.train_loader)
        for epoch in range(self.num_epochs):
            iter_count = 0

            epoch_time = time.time()
            self.model.train()
            for it, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                revin_layer = RevIN(num_features=self.input_c)
                x = revin_layer(input, 'norm')
                B, L, M = x.shape
                x_local_size = []
                x_patch_num = []
                for index, localsize in enumerate(self.local_size):
                    num = localsize + self.global_size[index]
                    result = []
                    front = num // 2
                    back = num - front
                    boundary = L - back
                    #in_size=0.0
                    for i in range(self.win_size):
                        if (i < front):
                            temp = x[:, 0, :].unsqueeze(1).repeat(1, front - i, 1)
                            temp1 = torch.cat((temp, x[:, 0:i, :]), dim=1)
                            temp1 = torch.cat((temp1, x[:, i:i + back, :]), dim=1)
                            result.append(temp1)
                        elif (i > boundary):
                            temp = x[:, L - 1, :].unsqueeze(1).repeat(1, back + i - L, 1)
                            temp1 = torch.cat((x[:, i - front:self.win_size, :], temp), dim=1)
                            result.append(temp1)
                        else:
                            temp = x[:, i - front:i + back, :].reshape(B, -1, M)
                            result.append(temp)
                    in_size = torch.cat(result, axis=0).reshape(L, B, num, M).permute(1, 0, 3, 2)
                    in_size = cal_similar(self.similar,in_size,x,num)
                    in_size = torch.softmax(in_size,dim=-1)

                    site = num//2

                    num1 = localsize
                    front = num1 // 2
                    back = num1 - front
                    in_x = in_size[:,:,site-front:site+back]
                    in_y = torch.cat((in_size[:,:,0:site-front],in_size[:,:,site+back:num]),dim=-1)
                    x_local_size.append(in_x)
                    x_patch_num.append(in_y)
                series_loss = 0.0
                prior_loss = 0.0
                area_local_loss = 0.0
                area_global_loss = 0.0
                series, prior, series_1, prior_1, area_local, area_global, sigma_local, sigma_global = self.model(x,x_local_size, x_patch_num,op,it,in_size)

                for u in range(len(prior)):
                    series_loss += self.criterion(series[u],x_local_size[u])
                    prior_loss += self.criterion(prior[u], x_patch_num[u])
                    area_local_loss += torch.sum(1+area_local[u])
                    area_global_loss += torch.sum(1+area_global[u]*2)
                series_loss = series_loss / len(prior)
                prior_loss =  prior_loss / len(prior)
                area_local_loss = area_local_loss / len(prior)
                area_global_loss =area_global_loss/ len(prior)
                loss = (series_loss+ prior_loss)*self.r+(area_local_loss+area_global_loss)*(1-self.r)#+con_global_loss+con_local_loss
                loss.backward()

                if (it + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - it)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                self.optimizer.step()

            print(
                "Epoch: {0}, Cost time: {1:.3f}s ".format(
                    epoch + 1, time.time() - epoch_time))
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        op = "test"
        norm_op = True
        print("norm_op:",norm_op)
        pattern = "mean"
        print("pattern:", pattern)
        # (1) stastic on the train set
        attens_energy = []
        for it, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)

            revin_layer = RevIN(num_features=self.input_c)
            x = revin_layer(input, 'norm')
            B, L, M = x.shape
            x_local_size = []
            x_patch_num = []
            for index, localsize in enumerate(self.local_size):
                num = localsize + self.global_size[index]
                result = []
                front = num // 2
                back = num - front
                boundary = L - back
                for i in range(self.win_size):
                    if (i < front):
                        temp = x[:, 0, :].unsqueeze(1).repeat(1, front - i, 1)
                        temp1 = torch.cat((temp, x[:, 0:i, :]), dim=1)
                        temp1 = torch.cat((temp1, x[:, i:i + back, :]), dim=1)
                        result.append(temp1)
                    elif (i > boundary):
                        temp = x[:, L - 1, :].unsqueeze(1).repeat(1, back + i - L, 1)
                        temp1 = torch.cat((x[:, i - front:self.win_size, :], temp), dim=1)
                        result.append(temp1)
                    else:
                        temp = x[:, i - front:i + back, :].reshape(B, -1, M)
                        result.append(temp)
                in_size = torch.cat(result, axis=0).reshape(L, B, num, M).permute(1, 0, 3, 2)
                in_size = cal_similar(self.similar, in_size, x, num)
                in_size = torch.softmax(in_size, dim=-1)

                site = num // 2

                num1 = localsize
                front = num1 // 2
                back = num1 - front
                in_x = in_size[:, :, site - front:site + back]
                in_y = torch.cat((in_size[:, :, 0:site - front], in_size[:, :, site + back:num]), dim=-1)
                x_local_size.append(in_x)
                x_patch_num.append(in_y)

            series, prior, _, _,  area_local, area_global, _, _ = self.model(x,x_local_size,x_patch_num,op,it,in_size)
            series_loss = 0.0
            prior_loss = 0.0

            for u in range(len(prior)):
                series_loss += torch.sum(self.criterion_keep(series[u], x_local_size[u]), dim=-1)
                prior_loss += torch.sum(self.criterion_keep(prior[u], x_patch_num[u]), dim=-1)

            if norm_op == True:
                metric = minmax_norm(series_loss + prior_loss)
            else:
                metric = series_loss + prior_loss

            metric = torch.softmax(metric, dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for it, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)

            revin_layer = RevIN(num_features=self.input_c)
            x = revin_layer(input, 'norm')
            B, L, M = x.shape
            x_local_size = []
            x_patch_num = []
            for index, localsize in enumerate(self.local_size):
                num = localsize + self.global_size[index]
                result = []
                front = num // 2
                back = num - front
                boundary = L - back
                for i in range(self.win_size):
                    if (i < front):
                        temp = x[:, 0, :].unsqueeze(1).repeat(1, front - i, 1)
                        temp1 = torch.cat((temp, x[:, 0:i, :]), dim=1)
                        temp1 = torch.cat((temp1, x[:, i:i + back, :]), dim=1)
                        result.append(temp1)
                    elif (i > boundary):
                        temp = x[:, L - 1, :].unsqueeze(1).repeat(1, back + i - L, 1)
                        temp1 = torch.cat((x[:, i - front:self.win_size, :], temp), dim=1)
                        result.append(temp1)
                    else:
                        temp = x[:, i - front:i + back, :].reshape(B, -1, M)
                        result.append(temp)
                in_size = torch.cat(result, axis=0).reshape(L, B, num, M).permute(1, 0, 3, 2)
                in_size = cal_similar(self.similar, in_size, x, num)
                in_size = torch.softmax(in_size, dim=-1)

                site = num // 2

                num1 = localsize
                front = num1 // 2
                back = num1 - front
                in_x = in_size[:, :, site - front:site + back]
                in_y = torch.cat((in_size[:, :, 0:site - front], in_size[:, :, site + back:num]), dim=-1)
                x_local_size.append(in_x)
                x_patch_num.append(in_y)
            series, prior, _, _, area_local, area_global, _, _ = self.model(x, x_local_size, x_patch_num, op, it, in_size)
            series_loss = 0.0
            prior_loss = 0.0

            for u in range(len(prior)):
                series_loss += torch.sum(self.criterion_keep(series[u], x_local_size[u]), dim=-1)
                prior_loss += torch.sum(self.criterion_keep(prior[u], x_patch_num[u]), dim=-1)

            if norm_op == True:
                metric = minmax_norm(series_loss + prior_loss)
            else:
                metric = series_loss + prior_loss
            metric = torch.softmax(metric, dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("anormly_ratio",self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for it, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)

            revin_layer = RevIN(num_features=self.input_c)
            x = revin_layer(input, 'norm')
            B, L, M = x.shape
            x_local_size = []
            x_patch_num = []
            for index, localsize in enumerate(self.local_size):
                num = localsize + self.global_size[index]
                result = []
                front = num // 2
                back = num - front
                boundary = L - back
                for i in range(self.win_size):
                    if (i < front):
                        temp = x[:, 0, :].unsqueeze(1).repeat(1, front - i, 1)
                        temp1 = torch.cat((temp, x[:, 0:i, :]), dim=1)
                        temp1 = torch.cat((temp1, x[:, i:i + back, :]), dim=1)
                        result.append(temp1)
                    elif (i > boundary):
                        temp = x[:, L - 1, :].unsqueeze(1).repeat(1, back + i - L, 1)
                        temp1 = torch.cat((x[:, i - front:self.win_size, :], temp), dim=1)
                        result.append(temp1)
                    else:
                        temp = x[:, i - front:i + back, :].reshape(B, -1, M)
                        result.append(temp)
                in_size = torch.cat(result, axis=0).reshape(L, B, num, M).permute(1, 0, 3, 2)
                in_size = cal_similar(self.similar, in_size, x, num)
                # in_size = torch.cat([in_size]*self.head).reshape(self.head,B,L,num).permute(1,2,0,3)
                in_size = torch.softmax(in_size, dim=-1)

                site = num // 2

                num1 = localsize
                front = num1 // 2
                back = num1 - front
                in_x = in_size[:, :, site - front:site + back]
                in_y = torch.cat((in_size[:, :, 0:site - front], in_size[:, :, site + back:num]), dim=-1)
                x_local_size.append(in_x)
                x_patch_num.append(in_y)


            series, prior, _, _, area_local, area_global, _, _ = self.model(x, x_local_size, x_patch_num, op, it,in_size)
            series_loss = 0.0
            prior_loss = 0.0

            for u in range(len(prior)):
                series_loss += torch.sum(self.criterion_keep(series[u], x_local_size[u]), dim=-1)
                prior_loss += torch.sum(self.criterion_keep(prior[u], x_patch_num[u]), dim=-1)

            if norm_op == True:
                metric = minmax_norm(series_loss + prior_loss)
            else:
                metric = series_loss + prior_loss


            metric = torch.softmax(metric, dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        matrix = [self.index]
        scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
        for key, value in scores_simple.items():
            matrix.append(value)
            print('{0:21} : {1:0.4f}'.format(key, value))

        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        np.savetxt('score.txt', test_energy, fmt='%f', delimiter='\n')
        np.savetxt('pred.txt', pred, fmt='%d', delimiter='\n')
        np.savetxt('fact.txt', gt, fmt='%d', delimiter='\n')
        np.savetxt('discrepancy.txt', (gt==pred).astype(int), fmt='%d', delimiter='\n')

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision,
                                                                                                   recall, f_score))

        if self.data_path == 'UCR' or 'UCR_AUG':
            import csv
            with open('result/' + self.data_path + '.csv', 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(matrix)

        return accuracy, precision, recall, f_score
