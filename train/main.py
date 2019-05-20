import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import soundfile as sf
from scipy.io import loadmat
from utils.util import get_alpha
from utils.pesq import pesq
from utils.stft_istft import STFT
import torch
from config import *
from net.module import CRNN
from load_data.SpeechDataLoad import SpeechDataset, SpeechDataLoader, FeatureCreator
from utils.loss_set import LossHelper
import torch.optim as optim
from config import *
from tensorboardX import SummaryWriter
import progressbar


def save(TIME):
    pkl_name = 'model_' + str(TIME) + '.pkl'
    torch.save(module, pkl_name)


def validation(path, net):
    net.eval()
    files = os.listdir(path)
    pesq_unprocess = 0
    pesq_res = 0
    bar = progressbar.ProgressBar(0, len(files))
    stft = STFT(filter_length=FILTER_LENGTH, hop_length=HOP_LENGTH)
    for i in range(len(files)):
        bar.update(i)
        with torch.no_grad():
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
            try:
                p1 = pesq('clean.wav', 'mix.wav', 16000)
                p2 = pesq('clean.wav', 'est.wav', 16000)
            except:
                print('wrong test item : ' + files[i])
                pass
            pesq_unprocess += p1[0]
            pesq_res += p2[0]

    bar.finish()
    net.train()
    return [pesq_unprocess / len(files), pesq_res / len(files)]


def train(net, optim, data_loader, loss_helper, epoch):
    global global_step
    writer = SummaryWriter(LOG_STORE)
    feature_creator = FeatureCreator()
    bar = progressbar.ProgressBar(0, train_data_set.__len__() // TRAIN_BATCH_SIZE)
    for i in range(epoch):
        bar.start()
        sum_loss = 0
        for batch_idx, batch_info in enumerate(data_loader.get_data_loader()):
            bar.update(batch_idx)
            mix_mag, label, frame_list = feature_creator(batch_info)
            optim.zero_grad()
            output = net(mix_mag)

            loss = loss_helper.mse_loss(output, label, frame_list)
            sum_loss += loss.item()
            loss.backward()
            optim.step()
            if batch_idx % 100 == 0 and batch_idx != 0:
                writer.add_scalar('Train/loss', sum_loss / 100, global_step)
                sum_loss = 0
            # 验证集
            if batch_idx % 1000 == 0:
                base_pesq, res_pesq = validation(VALIDATION_DATA_PATH, net)
                writer.add_scalar('Train/base_pesq', base_pesq, global_step)
                writer.add_scalar('Train/res_pesq', res_pesq, global_step)
            global_step += 1
        bar.finish()


if __name__ == '__main__':
    global_step = 0
    validation_data_set = SpeechDataset(root_dir=VALIDATION_DATA_PATH)
    validation_data_loader = SpeechDataLoader(data_set=validation_data_set,
                                              batch_size=VALIDATION_BATCH_SIZE,
                                              is_shuffle=True)
    train_data_set = SpeechDataset(root_dir=TRAIN_DATA_PATH)
    data_loader = SpeechDataLoader(data_set=train_data_set,
                                   batch_size=TRAIN_BATCH_SIZE,
                                   is_shuffle=True)

    module = CRNN()
    module = module.cuda(CUDA_ID[0])
    # loss_fun = torch.nn.MSELoss()
    optimizer = optim.Adam(module.parameters(), lr=LR)
    loss_helper = LossHelper()
    train(module, optimizer, data_loader, loss_helper, EPOCH)
