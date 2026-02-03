import datetime
import json
import os
import random
import re
import time

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer


end_of_text = '<|endoftext|>'


def download_training_data(word_cnt, dir):
    print('\nConnecting to Hugging Face...')

    # Load data in streaming mode to avoid downloading the entire dataset
    dataset = load_dataset('HuggingFaceFW/fineweb-edu', split = 'train', streaming = True)
    
    shuffled_dataset = dataset.shuffle()
    it = iter(shuffled_dataset)

    texts = []
    
    pattern = re.compile(r"^[A-Za-z0-9 \n\(\)'\,\.\?\!\$\-\;\:]+$")

    tic = time.time()
    
    n = 0
    while n < word_cnt:
        try:
            item = next(it)
        except StopIteration:
            print('Token count too high for this dataset.')
            break

        text = item['text'].strip()

        # Filter out short texts
        if(len(text) < 1500):
            continue

        # Filter out texts with a high ratio of special characters and numbers
        ratio_special = len(re.findall(r"[0-9()',.?!$;:-]", text)) / len(text)
        if ratio_special > 0.03:
            continue
        
        # Clean text
        if pattern.fullmatch(text):
            text = text.strip() + '\n' # \n marks the end of paragraphs, so add one after the last paragraph

            texts.append(text)
            n += len(text.split())

            toc = time.time()
            td = datetime.timedelta(seconds = int(toc - tic))
            
            time_str = str(td).replace(',', '')
            end_char = '\r' if n < word_cnt else '\n'

            print(f'{time_str} | Processing text: {(100 * n / word_cnt):.0f}%', end = end_char)

    # Create train/dev/test splits
    n = len(texts)

    train_data = texts[:int(0.96 * n)]
    val_data = texts[int(0.96 * n):int(0.98 * n)]
    test_data = texts[int(0.98 * n):]

    # Jumble up texts
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    os.makedirs(dir, exist_ok = True)
    with open(f'{dir}/train.txt', 'w') as file:
        file.write(end_of_text.join(train_data))
    
    with open(f'{dir}/val.txt', 'w') as file:
        file.write(end_of_text.join(val_data))
    
    with open(f'{dir}/test.txt', 'w') as file:
        file.write(end_of_text.join(test_data))

    print('Data downloaded successfully!')


def load_corpus(path):
    with open(path, 'r') as file:
        text = file.read()
    
    return text


def train_tokenizer(train_data, dir, vocab_size):
    print('\nTraining tokenizer...')
    
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator([train_data], vocab_size = vocab_size, min_frequency = 2, special_tokens = [end_of_text]) # Need to pass training data as a list

    os.makedirs(dir, exist_ok = True)
    tokenizer.save(f'{dir}/tokenizer.json')

    print('Tokenizer trained successfully!\n')


def jsonl2text(path):
    texts = []
    with open(path, 'r') as file:
        for line in file:
            data = json.loads(line)
            
            instruction = data.get('instruction', '')
            response = data.get('response', '')
            
            texts.append(f'Answer this question: {instruction.strip()}\n\n{response.strip()}\n\n')

    # Create train/dev/test splits
    n = len(texts)
    
    train_data = texts[:int(0.8 * n)]
    val_data = texts[int(0.8 * n):int(0.9 * n)]
    test_data = texts[int(0.9 * n):]

    # Jumble up texts
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    dir = os.path.dirname(path)
    
    os.makedirs(dir, exist_ok = True)
    with open(f'{dir}/train.txt', 'w') as file:
        file.write(end_of_text.join(train_data))
    
    with open(f'{dir}/val.txt', 'w') as file:
        file.write(end_of_text.join(val_data))
    
    with open(f'{dir}/test.txt', 'w') as file:
        file.write(end_of_text.join(test_data))