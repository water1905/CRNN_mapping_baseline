from net.module import CRNN
import numpy as np
import torch.nn as nn
import torch
import soundfile as sf
from stft_istft import STFT
from scipy.io import loadmat
import os

# np.random.seed(123)
# # conv2d
# # 输入尺寸：num(B),in_channels,h_in,w_in
# # 输出尺寸：num(B),out_channels,h_out,w_out
#
# # lstm
# # 输入尺寸：input(seq_len,batch,input_size)
# # ,h0(num_layers * num_directions, batch, hidden_size)
# # ,c0(num_layers * num_directions, batch, hidden_size)
# # num_directions: 方向，1为正向，2为反向
# # 输出尺寸：
# # output (seq_len, batch, hidden_size * num_directions)
# # ,h_n (num_layers * num_directions, batch, hidden_size)
# # ,c_n (num_layers * num_directions, batch, hidden_size)
# data_in = np.random.rand(10, 1, 1, 161)
# # 网络结构
# conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
# conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
# conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
# conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 2))
# conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 3), stride=(1, 2))
# LSTM1 = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=2)
#
# linear1 = nn.Linear(1024, 512)
# linear2 = nn.Linear(512, 256)
# linear3 = nn.Linear(256, 161)
#
# out1 = conv1(torch.Tensor(data_in))
# out2 = conv2(out1)
# out3 = conv3(out2)
# out4 = conv4(out3)
# # B,256,T,4
# # 10,256,1,4
# out5 = conv5(out4)
# # 暂存out5
# temp = out5
# # print(temp.shape)
# # reshape部分
# out5 = out5.permute(0, 2, 1, 3)
#
# # TODO
# out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)
# out5 = out5.permute(1, 0, 2)
# # print(out5.shape)
# # 1,10,1024
# h0 = torch.randn(2, out5.size()[1], 1024)
# c0 = torch.randn(2, out5.size()[1], 1024)
# output, (hn, cn) = LSTM1(out5, (h0, c0))
# # reshape部分
# # T,B,256,4 -> B,256,T,4
# # 1,10,256,4
# output = output.reshape(output.size()[1], output.size()[0], -1)
# # 10，256，1，4
# # B 256 T 4
# # output = output.permute(1, 2, 0, 3)
# l1 = linear1(output)
# l2 = linear2(l1)
# l3 = linear3(l2)
# print(l3)
# # B T F
# print(l3.shape)


model_speech = torch.load('/home/yangyang/PycharmProjects/CRNN_mapping_baseline/recovery/model_speech_2.pkl')
data_path = '/home/yangyang/userspace/data/CRNN_mapping_baseline/predict/data/'
res_path = '/home/yangyang/userspace/data/CRNN_mapping_baseline/predict/res/'

data_files = os.listdir(data_path)

for i in range(len(data_files)):
    data = loadmat(data_path + data_files[i])
    speech = data['speech']
    noise = data['noise']
    sf.write('/home/yangyang/PycharmProjects/CRNN_mapping_baseline/res/recovery/speech/' + data_files[i][:-4] + '_real_speech' + '.wav', speech, 16000)

    mix = speech + noise
    sf.write('/home/yangyang/PycharmProjects/CRNN_mapping_baseline/res/recovery/mix/' + data_files[i][:-4] + '_mix' + '.wav', mix, 16000)

    stft = STFT(filter_length=320, hop_length=160)

    # 预测
    # mix stft
    mix = torch.Tensor(mix)
    mix = mix.squeeze()
    mix = mix.reshape(1, mix.shape[0])
    mix = stft.transform(mix)

    temp = mix
    temp = stft.inverse(temp)
    temp = temp.transpose(1, 0)
    # 取得实部和虚部
    mix_real = torch.squeeze(mix[:, :, :, 0])
    mix_imag = torch.squeeze(mix[:, :, :, 1])
    # T,F 幅度谱
    mix_mag = torch.sqrt(mix_real ** 2 + mix_imag ** 2)

    # noise stft
    speech = torch.Tensor(speech)
    speech = speech.squeeze()
    speech = speech.reshape(1, speech.shape[0])
    speech = stft.transform(speech)
    # 取得实部和虚部
    speech_real = torch.squeeze(speech[:, :, :, 0])
    speech_imag = torch.squeeze(speech[:, :, :, 1])
    # T,F 幅度谱
    speech_mag = torch.sqrt(speech_real ** 2 + speech_imag ** 2)
    speech_mag = speech_mag.reshape(1, 1, speech_mag.size()[0], speech_mag.size()[1])
    out_mag = model_speech(speech_mag.cuda())
    out_mag = out_mag.squeeze()

    out_b = out_mag.cpu() * mix_imag / mix_mag
    out_a = out_mag.cpu() * mix_real / mix_mag

    speech_in = torch.stack([out_a, out_b], -1)
    speech_in = speech_in.reshape(1, speech_in.size()[0], speech_in.size()[1], speech_in.size()[2])

    res = stft.inverse(speech_in)
    res = res.transpose(1, 0)
    noise_ = temp - res
    # sf.write('/home/yangyang/PycharmProjects/CRNN_mapping_baseline/res/recovery/noise/' + data_files[i][:-4] + '_noise_predict' + '.wav', noise_.detach().numpy(), 16000)

    sf.write('/home/yangyang/PycharmProjects/CRNN_mapping_baseline/res/recovery/speech/' + data_files[i][:-4] + '__predict_speech' + '.wav', res.detach().numpy(), 16000)






# 使用output和out5 在 第一纬度上进行拼接
# # 10，512，1，4
# res = torch.cat((output, temp), 1)
# res1 = convT1(res)
# res1 = torch.cat((res1, out4), 1)
# res2 = convT2(res1)
# res2 = torch.cat((res2, out3), 1)
# res3 = convT3(res2)
# res3 = torch.cat((res3, out2), 1)
# res4 = convT4(res3)
# # print(res4.shape)
# res4 = torch.cat((res4, out1), 1)
#
# print("Hello")
# # B, 1, T, 161
# # 10, 1, 1, 161
# res5 = convT5(res4)
#
# result = res5.permute(0, 2, 1, 3)
# result = result.reshape(result.size()[0], result.size()[1], -1)
# print(result.shape)

# rnn = nn.LSTM(10, 20, 2)
# input = torch.randn(5, 3, 10)
# h0 = torch.randn(2, 3, 20)
# c0 = torch.randn(2, 3, 20)
# output, (hn, cn) = rnn(input, (h0, c0))
#
# print(output)
# out = convT1(out)
# print(out.shape)
# arr = np.random.rand(10, 2, 1, 4)
# arr = torch.Tensor(arr)
# arr = arr.permute(0, 2, 1, 3)
# arr = arr.reshape(10, 1, -1)
# arr = arr.numpy()
# arr = arr.reshape()
# print(arr.shape)
# print(arr[0].shape)

# crnn = CRNN_mapping_baseline()
# print(crnn)







# input = torch.randn(1, 16, 12, 12)
# downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
# # 限制网络结构 or 前馈时候使用output_size即upsample(h, output_size = input.size())
# upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=1)
# h = downsample(input)
# print(h.size())
# output = upsample(h)
# print(output.size())







