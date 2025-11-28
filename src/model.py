import torch
from torch import nn
from torchinfo import summary  # 打印参数
import config


class InputMethodModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # num_embedding,embedding_dim 将词表给他，他将每个词映射为一个向量
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.EMBEDDING_DIM)
        self.rnn = nn.RNN(input_size=config.EMBEDDING_DIM, hidden_size=config.HIDDEN_SIZE, batch_first=True,
                          bidirectional=True, num_layers=2)
        # 输出维度为此表的大小，在此之后输出一个概率值，取词表中概率最大的为预测值
        # (128*512)*(512*20000) = 128 * 20000
        self.linear = nn.Linear(in_features=2 * config.HIDDEN_SIZE, out_features=vocab_size)

    # 前向传播
    def forward(self, x):
        # x (batch_size,seq_len)
        embed = self.embedding(x)
        # embed (batch_size,seq_len,embedding_dim)
        batch_size = x.size(0)
        h0 = torch.randn((2 * 2, batch_size, config.HIDDEN_SIZE)).to('cuda')
        output, hn = self.rnn(embed, h0)
        # output (batch_size,seq_len,hidden_size)
        # hn (4,batch_size, hidden_size) 如果使用hidden需要挤掉第一个维度
        # last_hidden = hn[-1:, :, :] # 单层单向
        h1 = hn[-2, :, :] # 多层多项
        hn = hn[-1, :, :]
        result_stack = torch.cat([h1, hn], dim=1)
        res = self.linear(result_stack)
        # output  (batch_size,vocab_size)
        return res


if __name__ == '__main__':
    model = InputMethodModel(vocab_size=20000).to('cuda')

    # 创建随机 dummy 输入用于展示模型结构
    dummy_input = torch.randint(
        low=0,
        high=20000,
        size=(config.BATCH_SIZE, config.SEQ_LEN),
        dtype=torch.long,
        device='cuda'
    )

    # 打印模型摘要
    summary(model, input_data=dummy_input)
    # out = model(dummy_input)
    # print(out.shape)
# ==========================================================================================
# InputMethodModel                         [128, 20000]              --
# ├─Embedding: 1-1                         [128, 5, 128]             2,560,000
# ├─RNN: 1-2                               [128, 5, 256]             98,816
# ├─Linear: 1-3                            [128, 20000]              5,140,000
# ==========================================================================================
