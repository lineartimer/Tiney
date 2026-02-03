import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def log_train(path, batches, losses, norms):
    header = f"{'Batch':<10}{'Loss':<15}{'Norm':<15}\n"
    header += '-' * 32 + '\n'

    with open(path, 'w') as file:
        file.write(header)

        for i in range(len(batches)):
            log_entry = f'{batches[i]:<10}{losses[i]:<15.4f}{norms[i]:<15.4f}\n'
            file.write(log_entry)


def log_gen(path, texts):
    with open(path, 'w') as file:
        for i in range(len(texts)):
            file.write(f'Loss of {texts[i][0]:.4f}: {texts[i][1]}\n\n')


def plot(path, batches, losses):
    plt.figure(figsize = (10, 5))
    plt.plot(batches, losses, label = 'Loss', color = 'blue')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training')
    plt.legend()
    plt.grid(True)

    plt.savefig(path)

    plt.close()
