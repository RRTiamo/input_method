import pandas as pd

import config


def row():
    df = pd.read_json(config.PROCESS_DIR / 'train_dataset.jsonl', lines=True, orient='records')
    print(df.shape)


if __name__ == '__main__':
    row()
