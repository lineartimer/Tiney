import os
import sys

from corpus import download_training_data, jsonl2text, load_corpus, train_tokenizer
from model import Config


# RTX A4500
# scp -P <port number> ../data/corpus/val.txt root@<ip address>:/workspace/data/corpus


if __name__ == '__main__':
    ft_switch = 'finetune'
    d_switch = 'download'

    dir = '../data'

    if ft_switch in sys.argv or f'-{ft_switch}' in sys.argv or f'--{ft_switch}' in sys.argv:
        # Process data used for fine-tuning the model
        jsonl2text('../data/corpus/databricks-dolly-15k.jsonl')
    elif d_switch in sys.argv or f'-{d_switch}' in sys.argv or f'--{d_switch}' in sys.argv:
        # Download high-quality training data
        download_training_data(2e9, f'{dir}/corpus')

        # Train tokenizer
        config = Config()
        train_data = load_corpus('../data/corpus/val.txt')
        train_tokenizer(train_data, '../data', config.vocab_size)

        # Necessary, otherwise script won't exit (due to a Hugging Face process)
        os._exit(0)
    else:
        print('\nPlease provide either the -download or the -finetune switch.\n')
