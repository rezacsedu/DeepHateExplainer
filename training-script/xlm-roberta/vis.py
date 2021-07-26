import matplotlib.pyplot as plt
import torch

def display_acc_curves(history, title):
    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    plt.savefig(f'../DeepHateLingo/training-script/xlm-roberta/train_loss_his/{title}.png')
    plt.show()

def display_loss_curves(history, title):
    plt.plot(history['train_loss'], label='train loss')
    plt.plot(history['val_loss'], label='validation loss')
    plt.title('Training and Validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    plt.savefig(f'../DeepHateLingo/training-script/xlm-roberta/train_loss_his/{title}.png')
