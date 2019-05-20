import random
import pickle
import soundfile as sf
import scipy.io as sio
import scipy.signal as signal
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from settings.train import *
from torch.utils.data import Dataset, DataLoader
from torch.autograd.variable import *
from abc import ABCMeta, abstractmethod

EPSILON = 1e-7
NOISE_PARTS_NUM = 20
MAX_WAV_LEN = 7 * 16000

#
SPEECH_1_KEY = 'speech_1'
SPEECH_2_KEY = 'speech_2'
MIX_1_KEY = 'mix_1'
MIX_2_KEY = 'mix_2'
VAD_LABEL = 'vad_label'
NFRAMES_KEY = 'num_frames'

# dataset config
# tuning
SPEECH_ONLY_RATE = 0  # 0.05
NOISE_ONLY_RATE = 0  # 0.01

# float energy
SPEECH_MAX = 1.2
SPEECH_MIN = 0.5

# simulate when the large angle call(occlusion the second microphone) rate
OCCLUSION_RATE = 0.01

IS_MAX_INPUT = False

INTENSIFY_SPEECH_PATH = '/dev/shm/intensify/wav'
INTENSIFY_NOISE_PATH = '/mnt/raid/mem_cpy/intensify/noise/'


def gen_target_file_list(target_dir, target_ext='.wav'):
    l = []
    for root, dirs, files in os.walk(target_dir, followlinks=True):
        for f in files:
            f = os.path.join(root, f)
            ext = os.path.splitext(f)[1]
            ext = ext.lower()
            if ext == target_ext and '._' not in f:
                l.append(f)
    return l


class SMDatasetBase(Dataset):
    '''
    Basic class for genetating speech, mixture and vad info
    1. gen speech、mix
    2. intensify wav and noise
    3. vad info
    '''

    __metaclass__ = ABCMeta

    def __init__(self, wav_dir, noise_dir, snr_list=None, real_data_dir=None):
        self.wav_dir = wav_dir
        self.snr_list = snr_list
        self.real_data_dir = real_data_dir
        if self.real_data_dir is not None:
            self.real_data = self._list_real_data(self.real_data_dir)
        self.wav_list = gen_target_file_list(self.wav_dir)
        self.noise_only_rate = NOISE_ONLY_RATE

        self.noise_data_info = self._list_noise_and_snr(noise_dir)

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        wav_path = self.wav_list[idx]
        s, vad_label = self._read_wav(wav_path)
        n, snr = self._read_noise(noise_len=len(s))
        num_frame = (s.shape[0] - WIN_LEN) // WIN_OFFSET + 1
        if max(abs(s)) < 0.01:
            print('may be error')
            s = np.zeros_like(s)
            vad_label = np.zeros_like(vad_label)
        if self.noise_only_rate > 0 and self.noise_only_rate > random.random():
            noise_only = True
        else:
            noise_only = False
        if self.real_data_dir is not None:
            real_data = self._gen_real_data(s)
        else:
            real_data = None
        res = self._gen_mix(s, n, snr, vad_label, real_data, num_frame, noise_only)
        return res

    def _gen_mix(self, s, n, snr, vad_label, real_data, num_frame, noise_only):
        pass

    def _read_noise(self, noise_len):
        noise_data_max_index = len(self.noise_data_info) - 1
        noise_data_info = self.noise_data_info[random.randint(0, noise_data_max_index)]

        noise_data = noise_data_info[0]
        data_parts = noise_data.shape[0]
        data_len = noise_data.shape[1]
        snr = noise_data_info[1][random.randint(0, len(noise_data_info[1]) - 1)]

        n_0, n_1 = random.randint(0, data_parts - 1), random.randint(0, data_len - noise_len - 1)
        n = noise_data[n_0:n_0 + 1, n_1:n_1 + noise_len]
        while np.sum(np.abs(n)) < 1:
            n_0, n_1 = random.randint(0, data_parts - 1), random.randint(0, data_len - noise_len - 1)
            n = noise_data[n_0:n_0 + 1, n_1:n_1 + noise_len]
        n = np.squeeze(n).astype('float32')
        return n, snr

    def _list_real_data(self, real_data_dir):
        real_data = None
        for full_path in gen_target_file_list(real_data_dir, '.npy'):
            if os.path.isfile(full_path):
                tmp = np.load(full_path, mmap_mode='c')
                if real_data is None:
                    real_data = tmp
                else:
                    real_data = np.concatenate([real_data, tmp], axis=0)
        return real_data

    def _gen_real_data(self, s):
        real_data_len, slice_len = self.real_data.shape[0], s.shape[0]
        rdm_index = random.randint(0, real_data_len - slice_len - 1)
        slice_real_data = self.real_data[rdm_index:rdm_index + slice_len]
        return slice_real_data

    def _read_wav(self, wav_path):
        vad_label = np.load('{}{}'.format(os.path.splitext(wav_path)[0], '_vad.npy'))
        s, s_rate = sf.read(wav_path)
        if s_rate != 16000 or len(s) < 1600:
            raise ValueError('Invalid Sample Rate! Path: - {}'.format(wav_path))
        if len(s) > MAX_WAV_LEN:
            vad_on_index = np.where(vad_label == 1.0)[0]
            if len(vad_on_index) > 50:
                start_index = vad_on_index[50]
            else:
                print('file - {}, vad = 1 frames < 50, it maybe illigle file, please check it'.format(wav_path))
                start_index = 0
            start_index = max(start_index - 200, 0)
            vad_label = vad_label[start_index:start_index + MAX_WAV_LEN // 160 - 1]
            s = s[start_index * 160: start_index * 160 + MAX_WAV_LEN]
        return s.astype('float32'), vad_label.astype('float32')

    def _list_noise_and_snr(self, noise_path):
        noise_bin_ext = '.bin'
        noise_snr_ext = '.txt'
        noise_info = []
        for f in os.listdir(noise_path):
            full_path = os.path.join(noise_path, f)
            if os.path.isfile(full_path) and os.path.splitext(f)[1] == noise_bin_ext:
                snr_config_path = full_path.replace(noise_bin_ext, noise_snr_ext)
                snr_list = []
                with open(snr_config_path, 'r') as snr_config:
                    for snr_ in snr_config.readlines():
                        snr_list.append(int(snr_))
                if len(snr_list) > 0:
                    if self.snr_list is not None:
                        snr_list = self.snr_list
                    noise_np = np.memmap(full_path, dtype='float32', mode='c')
                    noise_np = noise_np[:noise_np.shape[0] // NOISE_PARTS_NUM * NOISE_PARTS_NUM]
                    noise_data = np.reshape(noise_np, (NOISE_PARTS_NUM, noise_np.shape[0] // NOISE_PARTS_NUM))
                    noise_info.append((noise_data, snr_list))
        assert len(noise_info) > 0, 'Not Found noise bin or snr list'
        return noise_info


class HeadPhoneDataset(SMDatasetBase):
    '''
    Basic class for genetating speech, mixture and vad info
    1. gen speech、mix
    2. intensify wav and noise
    3. vad info
    '''

    __metaclass__ = ABCMeta

    def __init__(self, wav_dir, noise_dir, trans_func_path, snr_list, real_data_dir=None):
        super(HeadPhoneDataset, self).__init__(wav_dir, noise_dir, snr_list, real_data_dir)
        self.wav_dir = wav_dir
        self.noise_snr_list = snr_list

        self.wav_list = gen_target_file_list(self.wav_dir)
        self.noise_only_rate = NOISE_ONLY_RATE
        self.trans = self._gen_rdm_trans(trans_func_path)

    def _apply_trans_func(self, s, tf):
        s1 = signal.fftconvolve(s, tf[0], mode='full')[:len(s)]
        s2 = signal.fftconvolve(s, tf[1], mode='full')[:len(s)]
        return s1, s2

    def _gen_rdm_trans(self, trans_func_path):
        with open(trans_func_path, 'rb') as f:
            trans_func = pickle.load(f)
        key_ls = []
        for key in trans_func.keys():
            if 'log' in key:
                key_ls.append(key)

        tf_ls = []
        for i in range(10000):
            index_1 = random.randint(0, len(key_ls) - 1)
            tf_dict = trans_func[key_ls[index_1]]
            tf1_ls = tf_dict['tf1']
            tf2_ls = tf_dict['tf2']
            index_21 = random.randint(0, tf1_ls.shape[0] - 1)
            index_22 = random.randint(0, tf2_ls.shape[0] - 1)
            tf = np.stack([tf1_ls[index_21], tf2_ls[index_22]])
            tf_ls.append(tf)
        return np.stack(tf_ls, axis=0)

    def _simlate_mics_data(self, s):
        index = random.randint(0, self.trans.shape[0] - 1)
        tf = self.trans[index]
        s1, s2 = self._apply_trans_func(s, tf)
        return s1, s2

    def __len__(self):
        return len(self.wav_list)

    def _gen_mix(self, s, n, snr, vad_label, real_data, num_frame, noise_only):
        if max(abs(s)) < 0.01:
            print('may be error')
            s = np.zeros_like(s)
            vad_label = np.zeros_like(vad_label)
        else:
            s = s / np.max(np.abs(s)) * 0.6
        s1, s2 = self._simlate_mics_data(s)
        if noise_only:
            alpha = 1 / np.max(np.abs(n))
            s = np.zeros_like(s)
            s1 = np.zeros_like(s1)
            s2 = np.zeros_like(s2)
            vad_label = np.zeros_like(vad_label)
        else:
            alpha = np.sqrt(np.sum(s1 ** 2.0) / (EPSILON + np.sum(n ** 2.0) * (10.0 ** (snr / 10.0))))
        mix1 = s1 + alpha * n
        mix2 = s2 + alpha * n

        rdm = random.uniform(SPEECH_MIN, SPEECH_MAX)
        s1 *= rdm * 3
        s2 *= rdm * 3
        mix1 *= rdm * 3
        mix2 *= rdm * 3

        s1 = s / (np.max(np.abs(s)) + EPSILON) * np.max(np.abs(s1))
        sample_vad = (np.reshape(np.tile(np.expand_dims(vad_label, 1), (1, 160)), -1)[:s1.size]).astype(np.long)
        if sample_vad.size < s1.size:
            sample_vad = np.concatenate((sample_vad, np.zeros(s1.size - sample_vad.size)), axis=0)

        sel_s1 = s1[sample_vad == 1]
        if np.sum(sel_s1 ** 2) < 0.01:
            c = 0
        else:
            c = np.sqrt(1.0 * sel_s1.size / np.sum(sel_s1 ** 2))
        s1 = s1 * c

        num_frame = (s1.shape[0] - WIN_LEN) // WIN_OFFSET + 1

        res_dict = {SPEECH_1_KEY: torch.from_numpy(s1.astype(np.float32)),
                    SPEECH_2_KEY: torch.from_numpy(s2.astype(np.float32)),
                    MIX_1_KEY: torch.from_numpy(mix1.astype(np.float32)),
                    MIX_2_KEY: torch.from_numpy(mix2.astype(np.float32)),
                    VAD_LABEL: torch.from_numpy(vad_label.astype(np.long)),
                    NFRAMES_KEY: num_frame}

        # sf.write('./s.wav', np.stack([s1, s2], axis=0).T, 16000)
        # sf.write('./mix.wav', np.stack([mix1, mix2], axis=0).T, 16000)
        return res_dict


class BatchDataLoaderBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, s_mix_dataset, batch_size, is_shuffle=False, workers_num=0):
        self.dataloader = DataLoader(s_mix_dataset, batch_size=batch_size, shuffle=is_shuffle,
                                     num_workers=workers_num,
                                     collate_fn=self.collate_fn)

    def get_dataloader(self):
        return self.dataloader

    @staticmethod
    @abstractmethod
    def gen_batch_data(zip_res):
        return zip_res

    @staticmethod
    def collate_fn(batch):
        batch.sort(key=lambda x: x[NFRAMES_KEY], reverse=True)
        keys = batch[0].keys()
        parse_res = []
        for item in batch:
            l = []
            for key in keys:
                l.append(item[key])
            parse_res.append(l)
        zip_res = list(zip(*parse_res))
        batch_dict = {}
        for i, key in enumerate(keys):
            data = list(zip_res[i])
            if isinstance(data[0], torch.Tensor):
                data = pad_sequence(data, batch_first=True)
            batch_dict[key] = data
        # batch_dict = BatchDataLoader.gen_batch_data(batch_dict)
        return SMBatchInfo(batch_dict)


class SMBatchInfo(object):
    def __init__(self, batch_dict):
        super(SMBatchInfo, self).__init__()
        s1 = batch_dict[SPEECH_1_KEY] if SPEECH_1_KEY in batch_dict else None
        s2 = batch_dict[SPEECH_2_KEY] if SPEECH_2_KEY in batch_dict else None
        self.s = torch.stack([s1, s2], dim=1)
        mix1 = batch_dict[MIX_1_KEY] if MIX_1_KEY in batch_dict else None
        mix2 = batch_dict[MIX_2_KEY] if MIX_2_KEY in batch_dict else None
        self.mix = torch.stack([mix1, mix2], dim=1)
        self.vad_label = batch_dict[VAD_LABEL] if VAD_LABEL in batch_dict else None

        self.nframes = batch_dict[NFRAMES_KEY] if NFRAMES_KEY in batch_dict else None
