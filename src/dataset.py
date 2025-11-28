import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import config


class InputMethodDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_json(data_path, lines=True, orient='records').to_dict(
            orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_tensor = torch.tensor(self.data[index]['input'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[index]['target'], dtype=torch.long)
        return input_tensor, target_tensor


def get_daataloader(train=True):
    path = 'test_dataset.jsonl' if train else 'train_dataset.jsonl'
    dataset = InputMethodDataset(config.PROCESS_DIR / path)  # 数据对象
    # DataLoader 的第一个参数必须是Dataset数据类型的数据
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)


if __name__ == '__main__':
    dataloader = get_daataloader()
    for input_tensor, target_tensor in dataloader:
        print(input_tensor.shape)  # torch.Size([128, 5])
        print(target_tensor.shape)  # torch.Size([128])
        break
