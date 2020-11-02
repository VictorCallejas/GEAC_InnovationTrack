# SOME AUXILIARY FUNCTION FOR CLEANER NOTEBOOKS
import numpy as np 
import torch 
import matplotlib.pyplot as plt 
import seaborn as sns


def top10_accuracy_scorer(gt_idx, top10_idx):
    
    aciertos = 0

    for arr, gt in zip(top10_idx,gt_idx):
        if gt in arr:
            aciertos+=1
            
    top_10_accuracy =  aciertos / len(gt_idx)
    return top_10_accuracy


#col_names = ['train_loss','train_acc','train_top10','dev_loss', 'dev_acc','dev_top10']
def make_plot(training_stats):

    sns.set(style='darkgrid')
    sns.set(font_scale=1.5)

    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Training stats')

    fig.set_size_inches(25, 10)

    ax1.plot(training_stats['train_loss'], 'b-o', label='training')
    ax1.plot(training_stats['dev_loss'], 'b-o', label='validation')

    ax1.set_title("Loss")
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss') 
    ax1.legend()

    ax2.plot(training_stats['train_top10'], 'b-o', label='training')
    ax2.plot(training_stats['dev_top10'], 'b-o', label='validation')

    ax2.set_title("Top 10 Acc")
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Top 10 Acc')
    ax2.legend()

    plt.show()

# seq, seq_id, vocab_size, tokenizer
def tokenize_worker(b):
    tmp = np.zeros(b[2])
    tokens = b[3].encode(b[0]).ids
    for tok in tokens:
        tmp[tok]+=1
    #tmp = tmp / len(b[0]) # Divide by sequence len
    return tmp,b[1]