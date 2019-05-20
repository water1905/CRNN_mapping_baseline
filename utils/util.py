import numpy as np
import os
import re
import torch
from torch.autograd import Variable


def expandWindow(data, left, right):
    data = data.detach().cpu().numpy()
    sp = data.shape
    idx = 0
    exdata = np.zeros([sp[0], sp[1], sp[2] * (left + right + 1)])
    for i in range(-left, right+1):
        exdata[:, :, sp[2] * idx : sp[2] * (idx + 1)] = np.roll(data, shift=-i, axis=1)
        idx = idx + 1
    return Variable(torch.FloatTensor(exdata)).cuda(CUDA_ID[0])


def context_window(data, left, right):
    sp = data.data.shape
    # exdata = torch.zeros(sp[0], sp[1], sp[2] * (left + right + 1)).cuda(CUDA_ID[0])
    exdata = torch.zeros(sp[0], sp[1], sp[2] * (left + right + 1))
    for i in range(1, left + 1):
        exdata[:, i:, sp[2] * (left - i) : sp[2] * (left - i + 1)] = data.data[:, :-i,:]
    for i in range(1, right+1):
        exdata[:, :-i, sp[2] * (left + i):sp[2]*(left+i+1)] = data.data[:, i:, :]
    exdata[:, :, sp[2] * left : sp[2] * (left + 1)] = data.data
    return Variable(exdata)


def read_list(list_file):
    f = open(list_file, "r")
    lines = f.readlines()
    list_sig = []
    for x in lines:
        list_sig.append(x[:-1])
    f.close()
    return list_sig


def gen_list(wav_dir, append):
    """使用正则表达式获取相应文件的list
    wav_dir:路径
    append:文件类型，eg: .wav .mat
    """
    l = []
    lst = os.listdir(wav_dir)
    lst.sort()
    for f in lst:
        if re.search(append, f):
            l.append(f)
    return l


def write_log(file, name, train, validate):
    message = ''
    for m, val in enumerate(train):
        message += ' --TRerror%i=%.3f ' % (m, val.data.numpy())
    for m, val in enumerate(validate):
        message += ' --CVerror%i=%.3f ' % (m, val.data.numpy())
    file.write(name + ' ')
    file.write(message)
    file.write('/n')


def get_alpha(mix, constant=1):
    """
    求得进行能量归一化的alpha值
    :param mix: 带噪语音的采样点的array
    :param constant: 一般取值为1，即使噪声平均每个采样点的能量在1以内
    :return: 权值c
    """
    # c = np.sqrt(constant * mix.size / np.sum(mix**2)), s *= c, mix *= c
    return np.sqrt(constant * mix.size / np.sum(mix ** 2))
