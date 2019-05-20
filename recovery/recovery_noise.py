import torch
import os
import librosa
from scipy.io import loadmat, savemat
import numpy as np
import torch.nn as nn
from stft_istft import STFT
import soundfile as sf


def write_mat():
    # train_data_path = '/mnt/raid/data/public/SPEECH_ENHANCE_DATA/tr/'
    new_train_data_path = '/home/yangyang/userspace/data/CRNN_mapping_baseline/9dB/'
    # files = os.listdir(train_data_path)
    files = os.listdir(new_train_data_path)

    for i in range(0, len(files)):
        mat = loadmat(new_train_data_path + files[i])
        speech = mat['speech']
        noise = mat['noise']
        mix = speech + noise

        module = torch.load('model_9.pkl')

        # 三者都是从文件里读进来的，一堆采样点组成的元组(xxx,1)
        # stft
        speech = librosa.stft(np.squeeze(speech), 320, 160)
        noise = librosa.stft(np.squeeze(noise), 320, 160)
        mix = librosa.stft(np.squeeze(mix), 320, 160)
        # 幅度谱
        y_mix = abs(mix)
        y_speech = abs(speech)
        y_noise = abs(noise)

        y_mix = y_mix.T
        y_mix = y_mix.reshape(y_mix.shape[0], y_mix.shape[1])
        y_mix = y_mix.reshape(1, 1, y_mix.shape[0], y_mix.shape[1])

        y_speech = y_speech.T
        y_speech = y_speech.reshape(y_speech.shape[0], y_speech.shape[1])
        y_speech = y_speech.reshape(1, 1, y_speech.shape[0], y_speech.shape[1])

        y_noise = y_noise.T
        y_noise = y_noise.reshape(y_noise.shape[0], y_noise.shape[1])
        y_noise = y_noise.reshape(1, 1, y_noise.shape[0], y_noise.shape[1])
        # y_amplitude = y_amplitude.permute(3, 1, 0, 2)

        out_mix, lstm_mix = module(torch.Tensor(y_mix).cuda())
        out_speech, lstm_speech = module(torch.Tensor(y_speech).cuda())
        out_noise, lstm_noise = module(torch.Tensor(y_noise).cuda())

        savemat('/home/yangyang/userspace/data/CRNN_mapping_baseline/res/9dB/' + files[i][:-4] + '.mat', {'lstm_out_mix': lstm_mix.detach().cpu().numpy(), 'lstm_out_speech': lstm_speech.detach().cpu().numpy(), 'lstm_out_noise': lstm_noise.detach().cpu().numpy()})

        # loss_fun = nn.MSELoss()
        # print(loss_fun(lstm_mix, lstm_noise))
        # print(loss_fun(lstm_mix, lstm_speech))
        # print(loss_fun(lstm_speech, lstm_noise))
        print('-------------------------------')


model_noise = torch.load('/home/yangyang/PycharmProjects/CRNN_mapping_baseline/recovery/model_noise_2.pkl')
data_path = '/home/yangyang/userspace/data/CRNN_mapping_baseline/predict/data/'
res_path = '/home/yangyang/userspace/data/CRNN_mapping_baseline/predict/res/'

data_files = os.listdir(data_path)

for i in range(len(data_files)):
    data = loadmat(data_path + data_files[i])
    speech = data['speech']
    noise = data['noise']
    sf.write('/home/yangyang/PycharmProjects/CRNN_mapping_baseline/res/recovery/noise/' + data_files[i][:-4] + '_real_noise' + '.wav', noise, 16000)
    mix = speech + noise
    stft = STFT(filter_length=320, hop_length=160)

    # 预测
    # mix stft
    mix = torch.Tensor(mix)
    mix = mix.squeeze()
    mix = mix.reshape(1, mix.shape[0])
    mix = stft.transform(mix)

    # 暂存mix
    temp = mix
    temp = stft.inverse(temp)
    temp = temp.transpose(1, 0)


    # 取得实部和虚部
    mix_real = torch.squeeze(mix[:, :, :, 0])
    mix_imag = torch.squeeze(mix[:, :, :, 1])
    # T,F 幅度谱
    mix_mag = torch.sqrt(mix_real ** 2 + mix_imag ** 2)

    # noise stft
    noise = torch.Tensor(noise)
    noise = noise.squeeze()
    noise = noise.reshape(1, noise.shape[0])
    noise = stft.transform(noise)
    # 取得实部和虚部
    noise_real = torch.squeeze(noise[:, :, :, 0])
    noise_imag = torch.squeeze(noise[:, :, :, 1])
    # T,F 幅度谱
    noise_mag = torch.sqrt(noise_real ** 2 + noise_imag ** 2)
    noise_mag = noise_mag.reshape(1, 1, noise_mag.size()[0], noise_mag.size()[1])
    out_mag = model_noise(noise_mag.cuda())
    out_mag = out_mag.squeeze()
    # 预测
    out_b = out_mag.cpu() * mix_imag / mix_mag
    out_a = out_mag.cpu() * mix_real / mix_mag

    noise_in = torch.stack([out_a, out_b], -1)
    noise_in = noise_in.reshape(1, noise_in.size()[0], noise_in.size()[1], noise_in.size()[2])

    res = stft.inverse(noise_in)
    res = res.transpose(1, 0)
    speech_ = temp - res

    # sf.write('/home/yangyang/PycharmProjects/CRNN_mapping_baseline/res/recovery/speech/' + data_files[i][:-4] + '_speech_predict' + '.wav', speech_.detach().numpy(), 16000)
    sf.write('/home/yangyang/PycharmProjects/CRNN_mapping_baseline/res/recovery/noise/' + data_files[i][:-4] + '_predict_noise' + '.wav', res.detach().numpy(), 16000)



