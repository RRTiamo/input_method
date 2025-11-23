import pandas as pd
from sklearn.model_selection import train_test_split

import config
from tokenizer import JiebaTokenizer


def build_dataset(sentences, tokenizer):
    # 将训练数据转换为id，没找到是使用未登录词
    index_train_dataset = [tokenizer.encode(sentence) for sentence in sentences]
    # print(index_train_dataset)
    # 根据滑动窗口构建数据集 [{'input':[1,2,3,4,5],'target':[6]}]
    dataset = []
    for sentence in index_train_dataset:
        for i in range(len(sentence) - config.SEQ_LEN):
            input = sentence[i:i + config.SEQ_LEN]
            target = sentence[i + config.SEQ_LEN]
            dataset.append({'input': input, 'target': target})

    return dataset


def process():
    """
    预处理数据
    """
    # 相对路径存在执行错误  避免数据过大，测试太慢sample(frac=0.1)
    data = pd.read_json(config.RAW_DIR / 'synthesized_.jsonl', orient='records', lines=True).sample(frac=0.1)
    # 读出每个对话的句子
    sentences = []
    for dialog in data['dialog']:
        for sentence in dialog:
            sentences.append(sentence.split('：')[1])

    # 划分数据集
    train_dataset, test_dataset = train_test_split(sentences, test_size=0.2)

    # 创建词表
    JiebaTokenizer.build_vocab(train_dataset, config.PROCESS_DIR / 'vocab.txt')
    # 建立分词器对象
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESS_DIR / 'vocab.txt')

    # 构建训练数据集 [{'input':[1,2,3,4,5],'target':[6]}]
    train_dataset_input_target = build_dataset(train_dataset, tokenizer)
    pd.DataFrame(train_dataset_input_target).to_json(config.PROCESS_DIR / 'train_dataset.jsonl', orient='records',
                                                     lines=True)
    # 构建并保存测试数据集
    test_dataset_input_target = build_dataset(test_dataset, tokenizer)
    pd.DataFrame(test_dataset_input_target).to_json(config.PROCESS_DIR / 'test_dataset.jsonl', orient='records',
                                                    lines=True)


if __name__ == '__main__':
    process()
