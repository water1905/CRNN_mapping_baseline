from scipy.io import loadmat
import torch
import numpy as np
import librosa
import soundfile as sf
from stft_istft import STFT


data_path = '/mnt/raid/data/public/SPEECH_ENHANCE_DATA/tr/S_51_02_destroyerengine_-5_0000164.mat'
data = loadmat(data_path)
speech = data['speech']
noise = data['noise']

mix = speech + noise

sf.write('real_speech_-5dB.wav', speech, 16000)

stft = STFT(filter_length=320, hop_length=160)
# speech stft
speech = torch.Tensor(speech)
speech = speech.squeeze()
speech = speech.reshape(1, speech.shape[0])
speech = stft.transform(speech)
# 取得实部和虚部
speech_real = torch.squeeze(speech[:, :, :, 0])
speech_imag = torch.squeeze(speech[:, :, :, 1])
# T,F 幅度谱
speech_mag = torch.sqrt(speech_real ** 2 + speech_imag ** 2)

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

speech_b = speech_mag * mix_imag / mix_mag
speech_a = speech_mag * mix_real / mix_mag

speech_in = torch.stack([speech_a, speech_b], -1)
speech_in = speech_in.reshape(1, speech_in.size()[0], speech_in.size()[1], speech_in.size()[2])

res = stft.inverse(speech_in)
res = res.transpose(1, 0)
sf.write('speech_-5dB.wav', res.numpy(), 16000)
# data = loadmat(data_path)
# module_noise = torch.load('/home/yangyang/PycharmProjects/CRNN_mapping_baseline/test/model_noise_2.pkl')
# module_speech = torch.load('/home/yangyang/PycharmProjects/CRNN_mapping_baseline/test/model_speech_2.pkl')
# stft = STFT(filter_length=320, hop_length=160)
#
# # 带噪语音的相位和幅度谱
# mix = librosa.stft(np.squeeze(data['speech']), 320, 160) + librosa.stft(np.squeeze(data['noise']), 320, 160)
# mix_temp = abs(mix)
#
# # 噪声相位谱
# noise = np.squeeze(data['noise'])
# noise = noise.reshape(1, noise.shape[0])
# noise = stft.transform(torch.Tensor(noise))
# # 取得实部和虚部
# noise_real = torch.squeeze(noise[:, :, :, 0])
# noise_imag = torch.squeeze(noise[:, :, :, 1])
# noise_in = torch.sqrt(noise_real ** 2 + noise_imag ** 2)
# # 转置
# # noise_in = noise_in.transpose(1, 0)
# noise_in = noise_in.reshape(1, 1, noise_in.size()[0], noise_in.size()[1])
# # 预测噪声的幅度谱B C T F
# out_noise, lstm = module_noise(noise_in.cuda())
# out_noise = out_noise.reshape(out_noise.size()[1], out_noise.size()[2])
#
#
# # 使用带噪语音的相位谱和预测的noise的幅度谱恢复noise
# noise_imag = out_noise.cpu().detach().numpy() * mix.imag/mix_temp
# noise_real = out_noise.cpu().detach().numpy() * mix.real/mix_temp
# res_noise = torch.stack([torch.Tensor(noise_imag), torch.Tensor(noise_real)], -1)
#
#
#
#
#
# # 对语音处理
# speech = np.squeeze(data['speech'])
# speech = speech.reshape(1, speech.shape[0])
# speech = stft.transform(torch.Tensor(speech))
# # 取得实部和虚部
# speech_real = torch.squeeze(speech[:, :, :, 0])
# speech_imag = torch.squeeze(speech[:, :, :, 1])
# speech_in = torch.sqrt(speech_real ** 2 + speech_imag ** 2)
#
# speech_in = speech_in.reshape(1, 1, speech_in.size()[0], speech_in.size()[1])
# # 预测噪声的幅度谱B C T F
# out_speech, lstm = module_speech(speech_in.cuda())
# out_speech = out_speech.reshape(out_speech.size()[1], out_speech.size()[2])
#
#
#
#
# # 预测语音的幅度谱
# x_speech = abs(speech)
# x_speech = x_speech.T
# x_speech = x_speech.reshape(1, 1, x_speech.shape[0], x_speech.shape[1])
# out_speech, lstm = module_speech(torch.Tensor(x_speech).cuda())
# out_speech = out_speech.reshape(out_speech.size()[1], out_speech.size()[2])
#
# # 恢复语音
# speech_imag = out_speech.cpu().detach().numpy() * mix.imag/mix_temp
# speech_real = out_speech.cpu().detach().numpy() * mix.imag/mix_temp
#
# res_speech = torch.stack([torch.Tensor(speech_imag), torch.Tensor(speech_real)], -1)
# res_speech = res_speech.reshape(1, res_speech.size()[1], res_speech.size()[0], res_speech.size()[2])
#
#
# out = stft.inverse(res_speech)
# print(out)
