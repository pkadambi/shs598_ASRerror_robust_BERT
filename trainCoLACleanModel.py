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

ASR_ERROR_LEVELS = [0.05, 0.1, 0.2, 0.3] # Must be either

#TODO: change this are to inputs via argparse
_MODEL = 'bert-base-uncased'
_NUM_CLASSES = 2
_ASR_ERROR_LEVEL = ASR_ERROR_LEVELS[0]
_BATCH_SIZE = 8 #parameter from huggingface glue tutorial (also limited to fit on GPU)
_COLA_DIR = './data/CoLA/'
_LEARNING_RATE = 5e-5
_DISTILLATION_ALPHA = 1.
_DISTILLATION_TEMPERATURE = 4
_N_EPOCHS = 3
_DEVICE = torch.device('cuda')
_MAX_GRAD_NORM = 1.0 #For gradient clipping
# Step 0: instantiate model and tokenizer
tokenizer = BertTokenizer.from_pretrained(_MODEL, do_lower_case=True)
model = ASRBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = _NUM_CLASSES,
                                         output_attentions=False, output_hidden_states=False)

# Step 1a. Load/Prepare/Tokenize CoLA data
train_data_path = os.path.join(_COLA_DIR, 'train.tsv')
dev_data_path = os.path.join(_COLA_DIR, 'dev.tsv')

df_train = pd.read_csv(train_data_path, delimiter='\t', header=None)
df_dev = pd.read_csv(dev_data_path, delimiter='\t', header=None)

names = ['Source', 'Label', 'LabelInfo', 'Sentence']


train_sentences = df_train[3]
dev_sentences = df_dev[3]

train_labels = torch.tensor(df_train[1])
dev_labels = torch.tensor(df_dev[1])

maxlen_dev = get_max_length(train_sentences, tokenizer)
maxlen_train = get_max_length(dev_sentences, tokenizer)

tr_input_ids, tr_attention_masks = tokenize_sentences(train_sentences, tokenizer)
dev_input_ids, dev_attention_masks = tokenize_sentences(dev_sentences, tokenizer)

#One line wrapper for dataset
train_dataset = TensorDataset(tr_input_ids, tr_attention_masks, train_labels)
dev_dataset = TensorDataset(dev_input_ids, dev_attention_masks, dev_labels)

# wandb.init()
train_dataloader = create_dataloader(train_dataset, dataSamplingStyle = 'Random', batch_size=_BATCH_SIZE)
dev_dataloader = create_dataloader(dev_dataset, dataSamplingStyle = 'Sequential', batch_size=_BATCH_SIZE)

# Step 1b. Fine-tune bert on clean CoLA data
model.to(_DEVICE)

optimizer = create_optimizer(model, learning_rate=_LEARNING_RATE)
scheduler = create_scheduler(train_dataloader, optimizer, nepochs=_N_EPOCHS)
# epochs = wandb.config.epochs
print(_N_EPOCHS)

model.train()
startime=time.time()

trloss_by_iter = []
for epoch in range(0, _N_EPOCHS):
    print('Started Epoch %d' % epoch)
    train_loss = 0

    for step, batch in enumerate(train_dataloader):
        if step % 50 == 0 and not step == 0:
            elapsed = time.time() - startime
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:.2f}.'.format(step, len(train_dataloader), elapsed))
        batch_input_id =  batch[0].to(_DEVICE)
        batch_atn_msk = batch[1].to(_DEVICE)
        batch_label = batch[2].to(_DEVICE)
        model.zero_grad()

        loss, logits = model(batch_input_id, attention_mask = batch_atn_msk, labels=batch_label, token_type_ids=None)
        batch_loss = loss.item()
        train_loss += batch_loss
        trloss_by_iter.append(batch_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), _MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        if step % 100==0:
            print('Train Loss {:.3f}'.format(batch_loss))

    trlossav = train_loss/len(train_dataloader)

    # pdb.set_trace()
    dev_loss_total = 0

    dev_accuracy, avg_dev_loss, dev_logits, dev_labels = dev_set_evaluate(model, dev_dataloader)
    mcc = evaluate_matthews_correlation(dev_logits,dev_labels)
    print('Dev Set MCC {:4f}'.format(mcc))



    # dev_logits = []
    # dev_labels = []
    # for batch in dev_dataloader:
    #     batch_input_id = batch[0].to(_DEVICE)
    #     batch_atn_mask = batch[1].to(_DEVICE)
    #     batch_label = batch[2].to(_DEVICE)
    #
    #     with torch.no_grad():
    #         loss, logits = model(batch_input_id, attention_mask=batch_atn_mask,  labels=batch_label, token_type_ids=None)
    #
    #     dev_loss_total+=loss
    #     logits = logits.detach().cpu().numpy()
    #     label_ids = batch_label.cpu().numpy()
    #     dev_logits.append(logits)
    #     dev_labels.append(label_ids)
    #
    # dev_logits = np.concatenate(dev_logits, axis=0)
    # dev_labels = np.concatenate(dev_labels)
    # total_dev_accuracy = flat_accuracy(dev_logits, dev_labels)
    # avg_dev_loss = dev_loss_total/len(dev_dataloader)
    # print('Dev Accuracy {:0.3f}'.format(total_dev_accuracy))
    # print('Total Dev Loss: {:0.3f} '.format(dev_loss_total))
    # print('Average Loss: {:0.3f}'.format(avg_dev_loss))






#TODO: generate a sequential data sampler on the training set in order to get the logits for the entire test set


# Step 2: Generate distillation ready dataset (utils function takes care of all of this)
pdb.set_trace()
# generate sequential dataloader on the train dataset
teacher_logits = generate_distillation_logits(train_dataset, model)


# evaluate the fine tuned model on the dataset
distillation_dataset = TensorDataset(tr_input_ids, tr_attention_masks, train_labels)


# save the output logits as the "teacher logits"

# create a new dataset where



# Step 3: Load/Prepare/Tokenize noisy CoLA data

#noisy_train_dataset = TensorDataset(train_sentences_noisy, tr_attention_masks_noisy, train_labels, teacher_soft_logits)
#noisy_dev dataset = TensorDataset(dev)




# Step 4:






#--------------------------------------------------
# Cross Validataion of distillation parameters
#--------------------------------------------------

