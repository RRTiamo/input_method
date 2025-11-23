from pathlib import Path

# __file__为当前文件的目录
ROOT_DIR = Path(__file__).parent.parent

PROCESS_DIR = ROOT_DIR / 'data' / 'processed'
RAW_DIR = ROOT_DIR / 'data' / 'raw'
LOGS_DIR = ROOT_DIR / 'logs'
MODELS_DIR = ROOT_DIR / 'models'

SEQ_LEN = 5
BATCH_SIZE = 128
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 10

if __name__ == '__main__':
    print(RAW_DIR)
