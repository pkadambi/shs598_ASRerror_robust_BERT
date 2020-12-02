# from distillationCompatibleBERT import ASRBertForSequenceClassification #DONT DO THIS circular import
import os
import pdb
import torch
import wandb
import numpy as np
import pickle as pkl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import matthews_corrcoef, f1_score
from scipy.stats import pearsonr, spearmanr

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    # pdb.set_trace()
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def get_max_length(sentences, tokenizer):
    max_len = 0
    for sent in sentences:
        input_ids = tokenizer.encode(sent, add_special_tokens=True)
        max_len = max(max_len, len(input_ids))
    return max_len

def create_dataloader(dataset, dataSamplingStyle, batch_size):
    '''
    :param dataset: the train/test/dev dataset
    :param dataSamplingStyle: Can be either 'Random', 'Sequential''
    :return: a dataloader for either the train/test/dev data wiht the specified data sampling style
    '''
    # batch_size = wandb.config.batch_size/
    print('batch_size = ', batch_size)

    if dataSamplingStyle.lower()=='random':
        sampler = RandomSampler(dataset)
    elif dataSamplingStyle.lower()=='sequential':
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader


def tokenize_sentences(sentences, tokenizer):
    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(sent, add_special_tokens=True, #Add special tokens adds [cls] and [sep] for stat end
                                             # max_length=64, padding=True,
                                             max_length=64, pad_to_max_length=True,
                                             return_attention_mask=True, return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    # pdb.set_trace()
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks


def create_optimizer(model, learning_rate, ADAMR=False):
    #TODO: add AdamR for fisher regularization
     # learning_rate = wandb.config.learning_ratefr
     optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
     return optimizer

def create_scheduler(dataloader, optimizer, nepochs, warmup_steps=0):
    #Learning rate scheduler (warmup steps, etc.) verified from huggingface
    # epochs = wandb.config.epochs
    total_steps = len(dataloader) * nepochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    return scheduler

def dev_set_evaluate(model, dev_dataloader, _DEVICE='cuda'):
    '''
    Tested for CoLA, evaluates a BertForSequenceClassification (or my distillation version of it) on an eval (dev) split
    :param model:
    :param dev_dataloader:
    :param _DEVICE:
    :return:
    '''
    model.eval()
    print('Calculating Dev Set Metrics')
    dev_loss_total = 0
    dev_logits = []
    dev_labels = []

    for batch in dev_dataloader:
        batch_input_id = batch[0].to(_DEVICE)
        batch_atn_mask = batch[1].to(_DEVICE)
        batch_label = batch[2].to(_DEVICE)

        with torch.no_grad():
            loss, logits = model(batch_input_id, attention_mask=batch_atn_mask,  labels=batch_label, token_type_ids=None)

        dev_loss_total+=loss
        logits = logits.detach().cpu().numpy()
        label_ids = batch_label.cpu().numpy()
        dev_logits.append(logits)
        dev_labels.append(label_ids)

    dev_logits = np.concatenate(dev_logits, axis=0)
    dev_labels = np.concatenate(dev_labels)
    dev_accuracy = flat_accuracy(dev_logits, dev_labels)
    avg_dev_loss = dev_loss_total/len(dev_dataloader)
    print('Total Dev Loss: {:0.3f} '.format(dev_loss_total))
    print('Average Loss: {:0.3f}'.format(avg_dev_loss))
    print('Dev Accuracy: {:0.3f}'.format(dev_accuracy))

    model.train()
    return dev_accuracy, avg_dev_loss, dev_logits, dev_labels


def evaluate_matthews_correlation(preds, labels):
    '''
    :param preds: the preds are logits [n_examples x 2]
    :param labels: the labels are [n_examples x 1]
    :return:
    '''
    preds_ = np.argmax(preds, axis=-1)

    labels_ = labels.copy()
    preds_[preds_<1] = -1       #need to code the negative example as -1
    labels_[labels_<1] = -1

    return matthews_corrcoef(labels_, preds_)


def evaluate_matthews_correlation_TEST():
    example_logits = np.array([[4, 5],[-2, 4],[-44, 5],[5, 4]])
    example_labels = np.array([1, -1, 1, 1])

    mcc=evaluate_matthews_correlation(example_logits, example_labels)

    print('logits')
    print(example_logits)
    print('labels')
    print(example_labels)
    print('MCC')
    print(mcc)


def generate_distillation_logits(trainDataset, trainedModel, device='cuda'):
    '''
    :param trainDataset: This is the original train dataset
    :param trainedModel: This is the trained model (acts as the teacher)
    :return:
    '''

    #must be sequential train data loader in order to line up properly with the training examples
    trainDataloaderSequential = create_dataloader(trainDataset, dataSamplingStyle='Sequential', batch_size=8)
    trainedModel.eval()

    distillationLogits = []
    print('Generating Logits on the Training Dataset')
    for step, batch in enumerate(trainDataloaderSequential):
        if step % 100 == 0:
            print('  Batch {:>5,}  of  {:>5,}'.format(step, len(trainDataloaderSequential)))
        batch_input_id = batch[0].to(device)
        batch_atn_mask = batch[1].to(device)
        batch_label = batch[2].to(device)

        loss, logits = trainedModel(batch_input_id, attention_mask = batch_atn_mask,
                                    labels=batch_label, token_type_ids=None)

        distillationLogits.append(logits.detach().cpu().numpy())

    distillationLogits = np.concatenate(distillationLogits, axis=0)

    trainedModel.train()

    return distillationLogits

def save_teacher_logits(teacher_logits, datasetPath):
    dct = {'teacher_logits':teacher_logits}
    pkl.dump(dct, open(os.path.join(datasetPath, 'teacherLogits.pkl'), 'wb'))
    print('Successfully saved teacher model logits to: ')
    print(os.path.join(datasetPath, 'teacherLogits.pkl'))

def load_teacher_logits(teacherLogitsPath):
    dct = pkl.load(open(teacherLogitsPath,'rb'))
    teacher_logits = dct['teacher_logits']
    return teacher_logits







