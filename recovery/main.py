import torch
from scipy.io import loadmat
import soundfile as sf
import os
from stft_istft import STFT


stft = STFT(filter_length=320, hop_length=160)
data_path = '/home/yangyang/userspace/data/CRNN_mapping_baseline/predict/data/'
files = os.listdir(data_path)

for i in range(len(files)):
    mat = loadmat(data_path + files[i])
    noise = mat['noise']
    speech = mat['speech']
    mix = noise + speech
    sf.write('/home/yangyang/PycharmProjects/CRNN_mapping_baseline/res/recovery/noise/real_noise_' + files[i] + '.wav', noise, 16000)
    # 预测
    # mix stft
    mix = torch.Tensor(mix)
    mix = mix.squeeze()
    mix = mix.reshape(1, mix.shape[0])
    mix = stft.transform(mix)
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
    noise_mag = noise_mag.reshape(noise_mag.size()[0], noise_mag.size()[1])

    # out_mag = model_noise(noise_mag.cuda())
    # out_mag = out_mag.squeeze()

    out_b = noise_mag * mix_imag / mix_mag
    out_a = noise_mag * mix_real / mix_mag

    noise_in = torch.stack([out_a, out_b], -1)
    noise_in = noise_in.reshape(1, noise_in.size()[0], noise_in.size()[1], noise_in.size()[2])

    res = stft.inverse(noise_in)
    res = res.transpose(1, 0)
    sf.write('/home/yangyang/PycharmProjects/CRNN_mapping_baseline/res/recovery/noise/noise_predict' + files[i] + '.wav', res.detach().numpy(), 16000)