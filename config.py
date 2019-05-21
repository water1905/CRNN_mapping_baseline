import os

cur_path = os.path.abspath(os.path.dirname(__file__))
PROJECT_NAME = os.path.split(cur_path)[1]

# 训练集：tr 验证集：cv 测试集：tt
# TYPE = 'tr'
# train
LR = 1e-3
EPOCH = 10
TRAIN_BATCH_SIZE = 32
TRAIN_DATA_PATH = '/mnt/raid/data/public/SPEECH_ENHANCE_DATA_NEW/tr/mix/'

# validation
VALIDATION_BATCH_SIZE = 1
VALIDATION_DATA_PATH = '/home/yangyang/userspace/data/envaluation_1/'

TEST_DATA_PATH = '/mnt/raid/data/public/SPEECH_ENHANCE_DATA_NEW/tt/mix/'

# model
MODEL_STORE = os.path.join('/home/yangyang/userspace/module_store/tmp/', PROJECT_NAME + '/')
if not os.path.exists(MODEL_STORE):
    os.mkdir(MODEL_STORE)
    print('Create model store file  successful!\n'
          'Path: \"{}\"'.format(MODEL_STORE))
else:
    print('The model store path: {}'.format(MODEL_STORE))

# log
LOG_STORE = os.path.join('/home/yangyang/userspace/log/tmp/', PROJECT_NAME + '/')
if not os.path.exists(LOG_STORE):
    os.mkdir(LOG_STORE)
    print('Create log store file  successful!\n'
          'Path: \"{}\"'.format(LOG_STORE))
else:
    print('The log store path: {}'.format(LOG_STORE))


# result
RESULT_STORE = os.path.join('/home/yangyang/userspace/result/', PROJECT_NAME + '/')
if not os.path.exists(RESULT_STORE):
    os.mkdir(RESULT_STORE)
    print('Create validation result store file  successful!\n'
          'Path: \"{}\"'.format(RESULT_STORE))
else:
    print('The validation result store path: {}'.format(RESULT_STORE))

FILTER_LENGTH = 320
HOP_LENGTH = 160

EPSILON = 1e-7
CUDA_ID = ['cuda:1']
