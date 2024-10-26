import torch
import torch.nn as nn
from einops import rearrange
from model.RevIN import RevIN
from tkinter import _flatten
import torch.nn.functional as F
import math
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 第一个全连接层
        self.fc2 = nn.Linear(hidden_size, output_size) # 第二个全连接层

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 使用ReLU作为激活函数
        x = self.fc2(x)
        return x

def calculate_area(mu, sigma, x):
    # 计算正态分布的累积分布函数 CDF
    dist = torch.distributions.Normal(mu, sigma)
    cdf1 = dist.cdf(x[:,:,0])
    cdf2 = dist.cdf(x[:,:,-1])

    return cdf1 - cdf2

def calculate_area_1(mu, sigma, x):
    # 计算正态分布的累积分布函数 CDF
    dist = torch.distributions.Normal(mu, sigma)
    cdf = dist.cdf(x[:,:,-1])

    return -cdf
class PPLAD(nn.Module):
    def __init__(self, batch_size,win_size, enc_in, c_out, d_model=256, local_size=[3, 5, 7], global_size=[1], channel=55,
                 d_ff=512, dropout=0.05, activation='gelu', output_attention=True, ):
        super(PPLAD, self).__init__()
        self.output_attention = output_attention
        self.local_size = local_size
        self.global_size = global_size
        self.channel = channel
        self.win_size = win_size
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.batch_size = batch_size
        self.sigma_projection_local = MLP(local_size[0] + global_size[0],d_model,1)


    def forward(self,x_in, in_size, in_num, op, it,in_x):
        local_out = []
        global_out = []
        local_score = []
        global_score = []
        area_local_out = []
        area_global_out = []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        B, L, M = x_in.shape

        sigma = self.sigma_projection_local(in_x).reshape(B, L)#b l num
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1

        x_site = torch.arange(self.win_size)
        x_site = x_site.unsqueeze(0).expand(B, -1).to(device)

        x = torch.abs(x_site.unsqueeze(-1) - x_site.unsqueeze(-2))


        for index, localsize in enumerate(self.local_size):
            num = localsize
            result = []
            result_1 = []
            front = num // 2
            back = num - front
            boundary = L - back
            for i in range(self.win_size):
                if (i < front):
                    temp = x_site[:,  0].unsqueeze(-1).repeat(1,  front - i)
                    temp1 = torch.cat((temp, x_site[:,  0:i]), dim=-1)
                    temp1 = torch.cat((temp1, x_site[:,  i:i + back]), dim=-1)
                    result_1.append(temp1)
                elif (i > boundary):
                    temp = x_site[:,  L - 1].unsqueeze(-1).repeat(1,  back + i - L)
                    temp1 = torch.cat((x_site[:,  i - front:self.win_size], temp), dim=-1)
                    result_1.append(temp1)
                else:
                    temp = x_site[:,  i - front:i + back].reshape(B,  -1)
                    result_1.append(temp)
            area_in_local = torch.cat(result_1, axis=-1).reshape(B, L, num)

            num_1 = self.global_size[index] + num
            result = []
            result_1 = []
            front_1 = num_1 // 2
            back_1 = num_1 - front_1
            boundary_1 = L - back_1
            for i in range(self.win_size):
                if (i < front_1):
                    temp = x[:,  i, 0].unsqueeze(-1).repeat(1,  front_1 - i)
                    temp1 = torch.cat((temp, x[:,  i, 0:i]), dim=-1)
                    temp1 = torch.cat((temp1, x[:,  i, i:i + back_1]), dim=-1)
                    result.append(temp1)

                    temp = x_site[:,  0].unsqueeze(-1).repeat(1,  front_1 - i)
                    temp1 = torch.cat((temp, x_site[:,  0:i]), dim=-1)
                    temp1 = torch.cat((temp1, x_site[:,  i:i + back_1]), dim=-1)
                    result_1.append(temp1[:,  0:front_1 - front])
                elif (i > boundary_1):
                    temp = x[:,  i, L - 1].unsqueeze(-1).repeat(1,  back_1 + i - L)
                    temp1 = torch.cat((x[:,  i, i - front_1:self.win_size], temp), dim=-1)
                    result.append(temp1)

                    temp = x_site[:,  L - 1].unsqueeze(-1).repeat(1,  back_1 + i - L)
                    temp1 = torch.cat((x_site[:,  i - front_1:self.win_size], temp), dim=-1)
                    result_1.append(temp1[:,  0:front_1 - front])
                else:
                    temp = x[:,  i, i - front_1:i + back_1].reshape(B, -1)
                    result.append(temp)

                    result_1.append(x_site[:,  i - front_1:i - front])
            all_site = torch.cat(result, axis=-1).reshape(B, L, num_1)
            area_in_global = torch.cat(result_1, axis=-1).reshape(B, L, -1)


            num = localsize + self.global_size[index]

            sigma_local = sigma.reshape(B, L, 1).repeat(1, 1, num)

            site = num // 2
            num1 = localsize
            front = num1 // 2
            back = num1 - front

            gaussian_kernel = 1.0 / (math.sqrt(2 * math.pi) * sigma_local) * torch.exp(
                -all_site ** 2 / (2 * sigma_local ** 2))
            torch.softmax(gaussian_kernel, dim=-1)
            gaussian_kernel_local = gaussian_kernel[:, :,  site - front:site + back]
            gaussian_kernel_global = torch.cat(
                (gaussian_kernel[:, :,  0:site - front], gaussian_kernel[:, :,  site + back:num]),
                dim=-1)
            local_out.append(gaussian_kernel_local), global_out.append(
                gaussian_kernel_global)


            area_local = calculate_area(x_site, sigma, area_in_local)
            area_global = calculate_area_1(x_site, sigma, area_in_global)
            area_local_out.append(area_local), area_global_out.append(area_global)
        local_out = list(_flatten(local_out))  # 3
        global_out = list(_flatten(global_out))  # 3
        local_score = list(_flatten(local_score))  # 3
        global_score = list(_flatten(global_score))  # 3
        area_local_out = list(_flatten(area_local_out))  # 3
        area_global_out = list(_flatten(area_global_out))  # 3
        if self.output_attention:
            return local_out, global_out, local_score, global_score, area_local_out, area_global_out, sigma, sigma
        else:
            return None

