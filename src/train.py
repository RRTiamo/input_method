import time

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_daataloader
import config
import model as m
from tokenizer import JiebaTokenizer


def train_one_epoch(dataset, model, loss_fn, optimizer, device):
    loss = 0
    model.train()
    for input, target in tqdm(dataset, desc="训练"):
        # 设置设备
        input = input.to(device)
        target = target.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        output = model(input)
        # 计算损失
        loss = loss_fn(output, target)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 累计损失
        loss += loss.item()

    return loss / len(dataset)


def train():
    # 转换设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 获取数据
    dataset = get_daataloader()  # 已经是处理好的数据
    # 构建分词器对象
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESS_DIR / 'vocab.txt')
    # 创建模型
    model = m.InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
    # 建立损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 建立优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # 创建日志写入器绘制图形 加入时间，防止图被覆盖
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime('%Y-%m-%d-%H-%M-%S'))
    # 开始训练
    # 定义一个初始的最大的损失值
    best_loss = float('inf')
    for epoch in range(1, config.EPOCHS + 1):
        print(f"==========Epoch{epoch}==========")
        # 训练一个轮次
        loss = train_one_epoch(dataset, model, loss_fn, optimizer, device)
        # 打印一次损失
        print(f'loss {loss}')
        # 绘图
        writer.add_scalar('Loss', loss, epoch)
        # 保存模型参数
        if loss < best_loss:
            best_loss = loss  # 最优损失比当前损失小，保存
            torch.save(model.state_dict(), config.MODELS_DIR / 'model.pt')
            print('保存模型成功')
        else:
            print('无需保存')


if __name__ == '__main__':
    train()
