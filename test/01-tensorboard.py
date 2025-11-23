from torch.utils.tensorboard import SummaryWriter

# 参数为目录，输出日志
writer = SummaryWriter(log_dir='./logs')

for i in range(100):
    writer.add_scalar('test', i + 1, i)
    writer.add_scalar('test01', i ** 2, i)

writer.close()
