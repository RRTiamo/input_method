import torch
import config
from predict import predict_batch
from dataset import get_daataloader
import model as m
from tokenizer import JiebaTokenizer


def evaluate_model(model, dataset, device):
    """
    模型评估
    :param model: 模型
    :param dataset: 数据
    :param device: 设备
    :return: top1_acc,top_5acc
    """
    total_count = 0
    top1_count = 0
    top5_count = 0
    for inputs, targets in dataset:
        inputs = inputs.to(device)
        # 批量预测
        top5_index = predict_batch(model, inputs)
        targets_list = targets.tolist()
        for target, top5_index in zip(targets_list, top5_index):
            total_count += 1
            if target == top5_index[0]:
                top1_count += 1
            if target in top5_index:
                top5_count += 1

    return top1_count / total_count, top5_count / total_count


def run_evaluate():
    """
    调用模型评估函数
    """
    dataset = get_daataloader(train=False)
    # 转换设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 构建分词器对象
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESS_DIR / 'vocab.txt')
    # 创建模型
    model = m.InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
    # 加载模型参数
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))

    top1_acc, top5_acc = evaluate_model(model, dataset, device)
    print("========== ACC ==========")
    print(f'top1_acc：{top1_acc:.4f}')
    print(f'top5_acc：{top5_acc:.4f}')
    print("=" * 10)


if __name__ == '__main__':
    run_evaluate()
