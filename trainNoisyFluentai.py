import pdb
import wandb
import datasets
import transformers
from distillationCompatibleBERT import *
from transformers import BertTokenizer
import pandas as pd
import os
from utils import *
# from trainer import *
from torch.utils.data import TensorDataset
import time as time

ASR_WER_LEVELS = [0.1] # Must be either
import argparse
parser = argparse.ArgumentParser(description='process distillation information')
parser.add_argument('--T', type=float, default=None)
args = parser.parse_args()

#TODO: change this are to inputs via argparse
_MODEL = 'bert-base-uncased'
_NUM_CLASSES = 31
_BATCH_SIZE = 8 #parameter from huggingface glue tutorial (also limited to fit on GPU)

_FLUENTAI_DIR = './data/fluentai/'
# get the labels
train_data_path = os.path.join(_FLUENTAI_DIR, 'train_data.csv')
dev_data_path = os.path.join(_FLUENTAI_DIR, 'valid_data.csv')
df_train = pd.read_csv(train_data_path)
df_dev = pd.read_csv(dev_data_path)
train_labels = torch.tensor(df_train['label'])
dev_labels = torch.tensor(df_dev['label'])

_CORRUPTED_DIR = './data/fluentaiCorrupted%.2f/'
_TEACHER_LOGITS_PATH = './data/fluentai/teacherLogits.pkl'
_LEARNING_RATE =1e-4

#post normal training distillation
_DISTILLATION_LEARNING_RATE = _LEARNING_RATE/2
_DISTILLATION_LR_WARMUP_STEPS = 0

_DISTILLATION_ALPHA = 1.
_DISTILLATION_TEMPERATURE = args.T
_DEVICE = torch.device('cuda')
_MAX_GRAD_NORM = 1.0 #For gradient clipping

if _DISTILLATION_TEMPERATURE is None:
    DF_NAME = './results/fluentai_baseline_wer_sweep.csv'
    _N_EPOCHS = 1
else:
    DF_NAME = './results/fluentai_distillation_temp%.1f.csv' % (_DISTILLATION_TEMPERATURE)
    #TODO: using warmup and more epochs to train with distillation especially at higher WER
    _N_EPOCHS = 1


results_df = pd.DataFrame()
results_df['WER'] = ASR_WER_LEVELS

DEV_SET_LOSSES = []
DEV_SET_MCCs = []
DEV_SET_ACCURACIES = []

for WER_LEVEL in ASR_WER_LEVELS:

    if _DISTILLATION_TEMPERATURE is not None:
        xEntLoss = []
        distilLoss = []

    print('****************************')
    print('**** BEGAN TRAINING FOR WER: {:.2f}'.format(WER_LEVEL))
    print('****************************')

    _CORRUPTED_DIR_WER = _CORRUPTED_DIR % WER_LEVEL
    # Step 0: instantiate model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(_MODEL, do_lower_case=True)
    model = ASRBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = _NUM_CLASSES,
                                             output_attentions=False, output_hidden_states=False,)
    model.dropout.p = 0.2  # will this fix the overfitting problem?
    model.alpha_distil = _DISTILLATION_ALPHA #Set distillation multiplier
    model.distil_T = _DISTILLATION_TEMPERATURE

    # Step 1a. Load/Prepare/Tokenize CoLA data
    train_data_path = os.path.join(_CORRUPTED_DIR_WER, 'corrupted_train_utterances.csv')
    dev_data_path = os.path.join(_CORRUPTED_DIR_WER, 'corrupted_valid_utterances.csv')

    df_train = pd.read_csv(train_data_path, header=None)
    df_dev = pd.read_csv(dev_data_path, header=None)
    # pdb.set_trace()
    train_sentences = df_train[0]
    dev_sentences = df_dev[0]

    maxlen_dev = get_max_length(train_sentences, tokenizer)
    maxlen_train = get_max_length(dev_sentences, tokenizer)

    tr_input_ids, tr_attention_masks = tokenize_sentences(train_sentences, tokenizer)
    dev_input_ids, dev_attention_masks = tokenize_sentences(dev_sentences, tokenizer)

    #One line wrapper for dataset

    if _DISTILLATION_TEMPERATURE is not None:
        teacher_logits = torch.Tensor(load_teacher_logits(_TEACHER_LOGITS_PATH))
        train_dataset = TensorDataset(tr_input_ids, tr_attention_masks, train_labels, teacher_logits)
    else:
        train_dataset = TensorDataset(tr_input_ids, tr_attention_masks, train_labels)

    dev_dataset = TensorDataset(dev_input_ids, dev_attention_masks, dev_labels)

    train_dataloader = create_dataloader(train_dataset, dataSamplingStyle = 'Random', batch_size=_BATCH_SIZE)
    dev_dataloader = create_dataloader(dev_dataset, dataSamplingStyle = 'Sequential', batch_size=_BATCH_SIZE)

    # Step 1b. Fine-tune bert on clean CoLA data
    model.to(_DEVICE)

    optimizer = create_optimizer(model, learning_rate=_LEARNING_RATE)

    if _DISTILLATION_TEMPERATURE is not None:
        scheduler = create_scheduler(train_dataloader, optimizer, nepochs=_N_EPOCHS, )
    else:
        scheduler = create_scheduler(train_dataloader, optimizer, nepochs=_N_EPOCHS)

    # epochs = wandb.config.epochs
    print(_N_EPOCHS)

    model.train()
    startime=time.time()

    trloss_by_iter = []
    for epoch in range(0, _N_EPOCHS):
        print('-------------Started Epoch %d' % (epoch+1))
        train_loss = 0
        distloss = None
        for step, batch in enumerate(train_dataloader):
            if step % 50 == 0 and not step == 0:
                elapsed = time.time() - startime
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:.2f}.'.format(step, len(train_dataloader), elapsed))
            batch_input_id =  batch[0].to(_DEVICE)
            batch_atn_msk = batch[1].to(_DEVICE)
            batch_label = batch[2].to(_DEVICE)
            model.zero_grad()

            if _DISTILLATION_TEMPERATURE is None:
                loss, logits = model(batch_input_id, attention_mask = batch_atn_msk, labels=batch_label, token_type_ids=None)
            else:
                # pdb.set_trace()
                batch_teacher_logits = batch[3].to(_DEVICE)
                loss, xent, distloss, logits = model(batch_input_id, attention_mask = batch_atn_msk, labels=batch_label,
                                     token_type_ids=None, teacher_logits = batch_teacher_logits)
                xEntLoss.append(xent.item())
                distilLoss.append(distloss.item())
                # pdb.set_trace()

            batch_loss = loss.item()
            train_loss += batch_loss
            trloss_by_iter.append(batch_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), _MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            if step % 100==0:
                if distloss is not None:
                    print('Xent: {:.3f}, Distil: {:.3f}, total: {:.3f}'.format(xent, distloss, batch_loss))
                    print('Train Loss (Distil) {:.3f}'.format(batch_loss))
                else:
                    print('Train Loss (Xent) {:.3f}'.format(batch_loss))


        trlossav = train_loss/len(train_dataloader)

        dev_loss_total = 0

        dev_accuracy, avg_dev_loss, dev_logits, dev_labels = dev_set_evaluate(model, dev_dataloader)
        print('Dev Set Accuracy {:.3f}'.format(dev_accuracy))

    DEV_SET_LOSSES.append(avg_dev_loss)
    DEV_SET_ACCURACIES.append(dev_accuracy)



'''
WRITE RESULTS CSV
'''
results_df['Dev Losses'] = DEV_SET_LOSSES
results_df['Dev Acc'] = DEV_SET_ACCURACIES
# results_df['Dev MCC'] = DEV_SET_MCCs
results_df.to_csv(DF_NAME)












