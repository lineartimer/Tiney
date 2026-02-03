import datetime
import math
import os
import random
import time
import sys

import torch
from tokenizers import Tokenizer

from corpus import end_of_text
from generate import generate
from model import Config, DecoderOnlyTransformer, export_model, get_device, load_model, model_size, save_model
from utils import log_gen, log_train, plot


# pip install -r requirements.txt
# watch -n 1 nvidia-smi


class DataLoader:
    def __init__(self, B, T, tokenizer_path, corpus_path):
        self.B = B
        self.T = T
        self.X = None
        self.Y = None

        self._corpus_path = corpus_path
        self._tokenizer = Tokenizer.from_file(tokenizer_path)

        # Estimate batches per epoch based on the first 100 lines of the corpus
        text = ''
        file_size = os.path.getsize(corpus_path)
        with open(corpus_path, 'r') as file:
            for _ in range(100):
                text += file.readline()
        
        tokens = self._tokenizer.encode(text).ids
        token_cnt = int(file_size * len(tokens) / len(text))

        self.batches_per_epoch = int(token_cnt / (B * T))

        self._file = open(corpus_path, 'r')
        self._buffer = []
        self._finished = False
    
    def __del__(self):
        if not self._file.closed:
            self._file.close()
    
    def next_batch(self):
        if self._finished:
            self.X = None
            self.Y = None

            self._file = open(self._corpus_path, 'r')
            self._buffer = []

            self._finished = False
        
        while(len(self._buffer) < self.B * self.T + 1):
            line = next(self._file, None)
            if line is not None:
                self._buffer += self._tokenizer.encode(line).ids
            else:
                self._file.close()
                self._finished = True

                return False # Discard last partial batch
        
        self._deliver_batch()

        return True
    
    def _deliver_batch(self):
        tokens = torch.tensor(self._buffer[:self.B * self.T + 1])
        self._buffer = self._buffer[self.B * self.T:] # Keep remaining tokens
        
        self.X = (tokens[:-1]).view(self.B, self.T) # Inputs
        self.Y = (tokens[1:]).view(self.B, self.T) # Targets
    

def get_lr(step, max_lr, min_lr, warmup_steps, total_steps):
    # Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    # Min learning rate
    if step > total_steps:
        return min_lr
    
    # Cosine decay down to min learning rate
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)


def train(model_config = Config(), batch_size = 64, max_lr = 2e-3, weight_decay = 0.15, beta1 = 0.95, beta2 = 0.99, from_scratch = True):
    start_time = time.time()
    print(f'\nTraining started: {datetime.datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")}')

    device = get_device()
    if device == 'cpu':
        hardware_info = device.upper()
    else:
        hardware_info = f'GPU ({device.upper()})'

    print(f'Hardware: {hardware_info}')

    training_data_path = '../data/corpus/train.txt'
    val_data_path = '../data/corpus/val.txt'
    tokenizer_path = '../data/tokenizer.json'
    model_path = '../data/lm.pth'
    log_path = '../data/log_train.txt'
    plot_path = '../data/loss.png'
    gen_path = '../data/log_gen.txt'

    T = model_config.context_size
    if device == 'cuda': # For NVIDIA A4500 (20 GB VRAM, 12 vCPU)
        B = batch_size
    else: # For a laptop (convenient for 8 GB of RAM)
        B = 2 # There's about a 20x speed difference between a MacBook Pro's GPU and a GPU made for AI compute
    
    train_loader = DataLoader(B = B, T = T, tokenizer_path = tokenizer_path, corpus_path = training_data_path)
    val_loader = DataLoader(B = B, T = T, tokenizer_path = tokenizer_path, corpus_path = val_data_path)

    torch.set_float32_matmul_precision('high') # Speed optimization
    
    if from_scratch:
        model = DecoderOnlyTransformer(model_config)
    else:
        model = load_model(model_path)

    model.to(device)

    print(f'Model size: {model.size() / 1e6:.2f}M parameters')
    print(f'Training data: {(train_loader.batches_per_epoch * B * T)/1e6:.2f}M tokens')

    batches = []
    losses = []
    norms = []
    texts = []

    checkpoint_interval = 500

    min_lr = max_lr * 0.1 # The lowest value that seems optimal
    total_steps = train_loader.batches_per_epoch
    warmup_steps = int(total_steps * 0.01) # 1% warmup

    optimizer = model.configure_optimizers(max_lr, weight_decay, beta1, beta2, device)

    final_train_loss = float('inf')
    best_val_loss = float('inf')
    val_loss = 0
    no_improvement = False
    epoch = 0
    while not no_improvement:
        step = 0
        while train_loader.next_batch():
            tic = time.time()

            x = train_loader.X.to(device)
            y = train_loader.Y.to(device)

            optimizer.zero_grad() # Forgetting to zero out gradients is a common bug
            with torch.autocast(device_type = device, dtype = torch.bfloat16): # Speed optimization
                _, loss = model(x, y)

            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            lr = get_lr(step, max_lr, min_lr, warmup_steps, total_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            optimizer.step()

            # Wait for GPU to finish work
            if device == 'cuda':
                torch.cuda.synchronize()
            elif device == 'mps':
                torch.mps.synchronize()

            toc = time.time()
            tokens_per_sec = int(train_loader.B * train_loader.T / (toc - tic))
            
            # Log and plot training progression
            batches.append(batches[-1] + 1 if len(batches) > 0 else 1)
            losses.append(loss.item())
            norms.append(norm)
            
            log_train(log_path, batches, losses, norms)
            plot(plot_path, batches, losses)

            # Generate text and save model
            if step % checkpoint_interval == 0:
                prompt = f'{end_of_text}Answer this question: What is a car?'
                text = generate(model, prompt)
                texts.append((loss, f'{prompt}{text}'))

                log_gen(gen_path, texts)

                model.train()
                
                checkpoint = {
                    'batch': step,
                    'loss': loss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }

                save_model(checkpoint, model_path)
            
            time_elapsed = datetime.timedelta(seconds = int(toc - start_time))
            
            epoch_str = f' | Epoch: {epoch + 1}' if not from_scratch else ''
            print(f'\r{str(time_elapsed).replace(",", "")} {tokens_per_sec} tok/s{epoch_str} | Batch: {step + 1} | Loss: {loss.item():.2f} | Lr: {lr:.4e}', end = '')

            step += 1
        
        final_train_loss = loss.item()

        # If training from scratch, train for one epoch only (there's more than enough data for small models like this)
        if from_scratch:
            break
        else:
            model.eval()

            val_loss = 0
            step = 0
            with torch.no_grad():
                while val_loader.next_batch():
                    x = val_loader.X.to(device)
                    y = val_loader.Y.to(device)

                    with torch.autocast(device_type = device, dtype = torch.bfloat16): # Speed optimization
                        _, loss = model(x, y)
                    
                    val_loss += loss
                    step +=1
            
            val_loss /= step
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            else:
                no_improvement = True
            
            model.train()
        
        epoch += 1
        
    print('\n')

    return final_train_loss


if __name__ == '__main__':
    ft_switch = 'finetune'

    if ft_switch in sys.argv or f'-{ft_switch}' in sys.argv or f'--{ft_switch}' in sys.argv:
        # Finetune
        train(max_lr = 1e-3, from_scratch = False) # A lower learning rate works better for fine-tuning. (It also helps avoid 'catastrophic forgetting'.)

        # Export model in onnx format
        export_model('../data/lm.pth')
    else:
        # Pretrain
        train()
