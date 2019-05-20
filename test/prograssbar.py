import numpy as np
from tensorboardX import SummaryWriter

writer = SummaryWriter()

for epoch in range(100):
    # 保存的图的名字，Y轴数据，X轴数据（PS当Y轴数据不止一个时，使用add_scalars()）
    writer.add_scalar('scale/test', np.random.rand(), epoch)
    # writer.add_scalars('scale/scale_tests', {'xsin': epoch * np.sin(epoch), 'xcos': epoch * np.cos(epoch)}, epoch)
writer.close()


