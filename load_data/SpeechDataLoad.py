import os
import torch
import numpy as np
import librosa
import torch.nn as nn
from scipy.io import loadmat
from utils.util import get_alpha
from utils.stft_istft import STFT
from torch.utils.data import DataLoader, Dataset
from config import *
from utils.label_set import LabelHelper


class SpeechDataLoader(object):
    def __init__(self, data_set, batch_size, is_shuffle=True, num_workers=0):
        """
        初始化一个系统的Dataloader，只重写他的collate_fn方法
        :param data_set: 送入网络的data,dataset对象
        :param batch_size: 每次送入网络的data的数量，即多少句话
        :param is_shuffle: 是否打乱送入网络
        :param num_workers: dataloader多线程工作数，一般我们取0
        """
        self.data_loader = DataLoader(dataset=data_set,
                                      batch_size=batch_size,
                                      shuffle=is_shuffle,
                                      num_workers=num_workers,
                                      collate_fn=self.collate_fn)

    # 静态方法，由类和对象调用
    # 该函数返回对数据的处理，返回target,load_data
    @staticmethod
    def collate_fn(batch):
        """
        将每个batch中的数据pad成一样长，采取补零操作
        切记为@staticmethod方法
        :param batch: input和label的list
        :return:input、label和真实帧长 的list
        """
        mix_list = []
        speech_list = []
        noise_list = []
        frame_size_list = []
        for item in batch:
            # (T,F)
            mix_list.append(item[0])
            speech_list.append(item[1])
            noise_list.append(item[2])
            # 储存每句话的真实帧长，时域信息，用于计算loss
            frame_size_list.append(item[3])
        mix_list = nn.utils.rnn.pad_sequence(mix_list)
        speech_list = nn.utils.rnn.pad_sequence(speech_list)
        noise_list = nn.utils.rnn.pad_sequence(noise_list)

        mix_list = mix_list.permute(1, 0, 2)
        speech_list = speech_list.permute(1, 0, 2)
        noise_list = noise_list.permute(1, 0, 2)

        # data_list = (B,in_c,T,F)
        # target_list = (B,T,F)
        return BatchInfo(mix_list, speech_list, noise_list, frame_size_list)

    def get_data_loader(self):
        """
        获取Dataloader
        :return: dataloader对象
        """
        return self.data_loader


class SpeechDataset(Dataset):

    def __getitem__(self, index):
        """
        对于每个送入网络的数据进行处理
        PS：一般只对数据进行时域上的操作，其他操作如：STFT，送入CUDA之后进行
        :param index:送入网络数据的索引，一般是文件的索引
        :return:输入数据，相应的label
        """
        # 迭代输出需要的文件
        try:
            data = loadmat(self.root_dir + self.files[index])
        except:
            data = loadmat(self.root_dir + self.files[np.random.randint(0, 90000)])
        # 三者都是从文件里读进来的，一堆采样点组成的元组(xxx,1)
        speech = data['speech']
        noise = data['noise']
        mix = speech + noise

        c = get_alpha(mix)
        speech *= c
        mix *= c

        nframe = (len(speech) - FILTER_LENGTH) // HOP_LENGTH + 1
        # len_speech = nframe * HOP_LENGTH
        # speech = speech[:len_speech, :]
        # noise = noise[:len_speech, :]
        # mix = mix[:len_speech, :]
        # # stft
        # speech_ = self.stft.transform(torch.Tensor(speech.T))
        # mix_ = self.stft.transform(torch.Tensor(mix.T))
        # noise_ = self.stft.transform(torch.Tensor(noise.T))
        # # (B,T,F)
        # mix_real = mix_[:, :, :, 0]
        # mix_imag = mix_[:, :, :, 1]
        #
        # noise_real = noise_[:, :, :, 0]
        # noise_imag = noise_[:, :, :, 1]
        #
        # # mix_mag(T,F)
        # mix_mag = torch.sqrt(mix_imag ** 2 + mix_real ** 2).squeeze()
        # noise_mag = torch.sqrt(noise_imag ** 2 + noise_real ** 2).squeeze()

        # mix_mag(T,F)
        # PSM(T,F)
        return torch.Tensor(mix), torch.Tensor(speech), torch.Tensor(noise), nframe

    def __len__(self):
        """
        返回总体数据的长度
        :return: 总体数据的长度
        """
        return len(self.files)

    def __init__(self, root_dir):
        """
        初始化dataset，读入文件的list
        :param root_dir: 文件的根目录
        :param type: 暂时未用
        :param transform: 暂时未用
        """
        # 初始化变量
        self.stft = STFT(filter_length=320, hop_length=160)
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)

    def cal_IRM(self, s, n):
        """
        计算IRM
        :param s:纯净语音，tensor
        :param n:噪声：tensor
        :return: IRM
        """

        return torch.pow(s.pow(2) / (s.pow(2) + n.pow(2)), 0.5)

    def cal_PSM(self, s, y):
        """
        计算PSM
        :param s: 纯净语音，complex，即librosa.stft(s)
        :param y: 带噪语音，complex，即librosa.stft(s + n)
        :return: clip到（0，1）,shape(T,B,F)
        """
        # TODO 除0检测
        s_real = s[:, :, :, 0]
        s_imag = s[:, :, :, 1]
        y_real = y[:, :, :, 0]
        y_imag = y[:, :, :, 1]
        return ((s_real * y_real + s_imag * y_imag) / (y_real ** 2 + y_imag ** 2)).clamp(0, 1).squeeze()


class BatchInfo(object):

    def __init__(self, mix, speech, noise, nframe):
        self.mix = torch.Tensor(mix).cuda()
        self.speech = torch.Tensor(speech).cuda()
        self.noise = torch.Tensor(noise).cuda()
        self.nframe = nframe


class FeatureCreator(nn.Module):

    def __init__(self):
        super(FeatureCreator, self).__init__()
        self.stft = STFT(FILTER_LENGTH, HOP_LENGTH)
        self.label_helper = LabelHelper()

    def forward(self, batch_info):
        mix_spec = self.stft.transform(batch_info.mix)
        speech_spec = self.stft.transform(batch_info.speech)
        noise_spec = self.stft.transform(batch_info.noise)

        mix_real = mix_spec[:, :, :, 0]
        mix_imag = mix_spec[:, :, :, 1]

        mix_mag = torch.sqrt(mix_real ** 2 + mix_imag ** 2)
        mix_mag = mix_mag.unsqueeze(1)
        # 防止除零
        label = self.label_helper(speech_spec, noise_spec)
        # 验证label

        return mix_mag, label, batch_info.nframe




