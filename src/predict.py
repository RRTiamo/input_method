import torch

import config
import model as m
from tokenizer import JiebaTokenizer


def predict_batch(model, input_tensor):
    # 模型预测
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        # output (batch_size,vocab_size)
        # 取出前五个
        # dim 默认为最后一个维度
        top5_index = torch.topk(output, k=5).indices  # top_index (batch_size,5)
    top5_index_list = top5_index.tolist()
    return top5_index_list


def predict(text, tokenizer, model, device):
    """
    模型预测
    :param device: 设备
    :param tokenizer: 分词器对象
    :param model: 模型
    :param text: 输入数据
    :return: 前五个词
    """
    # 处理输入数据
    text_list_index = tokenizer.encode(text)
    input_tensor = torch.tensor([text_list_index]).to(device)  # input_tensor (batch_size,seq_len)

    # 批量预测
    top5_index = predict_batch(model, input_tensor)

    # 转换为词
    top5_words = [tokenizer.index2word[index] for index in top5_index[0]]
    return top5_words


def run_predict():
    # 转换设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 构建分词器对象
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESS_DIR / 'vocab.txt')

    # 创建模型
    model = m.InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
    # 加载模型参数
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))

    history_input = ''
    print('输入quit退出')
    while True:
        text = input('请输入>>>')
        if text in ['q', 'quit']:
            break
        if text.strip() == '':
            print('请输入下一个词')
            continue
        history_input += text
        print(history_input)
        words = predict(history_input, tokenizer, model, device)
        print(words)


if __name__ == '__main__':
    run_predict()
