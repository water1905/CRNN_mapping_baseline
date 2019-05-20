import os
from config import *
from scipy.io import loadmat


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
    files = os.listdir(VALIDATION_DATA_PATH)
    for item in files:
        file_name = item.split('_')
        db = file_name[1]
        noise_type = file_name[2]
        data_dic[noise_type] += 1
    print(data_dic)
        # data = loadmat(VALIDATION_DATA_PATH + item)
