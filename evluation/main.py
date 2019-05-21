import sys
import os
from scipy.io import loadmat
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from config import *
from scipy.io import loadmat
import progressbar
from utils.stft_istft import STFT
from utils.util import get_alpha
from utils.pesq import pesq
import torch
import soundfile as sf
from net.module import CRNN


def cal_pesq(net, path):
    files = os.listdir(path)
    pesq_unprocess = 0
    pesq_res = 0
    bar = progressbar.ProgressBar(0, len(files))
    stft = STFT(filter_length=FILTER_LENGTH, hop_length=HOP_LENGTH)
    time = 0
    for i in range(len(files)):
        bar.update(i)
        with torch.no_grad():
            if files[i].split('_')[1] == '0':
                time += 1
                speech = loadmat(path + files[i])['speech']
                noise = loadmat(path + files[i])['noise']
                mix = speech + noise

                sf.write('clean.wav', speech, 16000)
                sf.write('mix.wav', mix, 16000)

                c = get_alpha(mix)
                mix *= c
                speech *= c
                noise *= c

                speech = stft.transform(torch.Tensor(speech.T).cuda())
                mix = stft.transform(torch.Tensor(mix.T).cuda())
                noise = stft.transform(torch.Tensor(noise.T).cuda())

                mix_real = mix[:, :, :, 0]
                mix_imag = mix[:, :, :, 1]
                mix_mag = torch.sqrt(mix_real ** 2 + mix_imag ** 2)

                # mix_(T,F)
                mix_mag = mix_mag.unsqueeze(0).cuda()
                # output(1, T, F)

                mapping_out = net(mix_mag)

                res_real = mapping_out * mix_real / mix_mag.squeeze(0)
                res_imag = mapping_out * mix_imag / mix_mag.squeeze(0)

                res = torch.stack([res_real, res_imag], 3)
                output = stft.inverse(res)

                output = output.permute(1, 0).detach().cpu().numpy()

                # 写入的必须是（F,T）istft之后的
                sf.write('est.wav', output / c, 16000)
                p1 = pesq('clean.wav', 'mix.wav', 16000)
                p2 = pesq('clean.wav', 'est.wav', 16000)

                pesq_unprocess += p1[0]
                pesq_res += p2[0]

    bar.finish()
    net.train()
    return [pesq_unprocess / time, pesq_res / time]

data_dic = {
    'babble': 0,
    'destroyerengine': 0,
    'f16': 0,
    'buccaneer1': 0,
    'factory1': 0,
    'factory2': 0,
    'destroyerops': 0
}

if __name__ == '__main__':
    res = CRNN()
    for i in range(1, 2):
        file_handle = open('res_' + str(i) + '.txt', mode='w')
        res = torch.load(MODEL_STORE + 'model_' + str(i) + '.pkl')
        base_pesq, res_pesq = cal_pesq(res, VALIDATION_DATA_PATH)
        print(base_pesq)
        print(res_pesq)
        file_handle.write(str(base_pesq))
        file_handle.write('\n')
        file_handle.write(str(res_pesq))
        file_handle.close()
    # data = loadmat(VALIDATION_DATA_PATH + item)
