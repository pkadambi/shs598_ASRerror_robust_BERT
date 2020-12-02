import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


#KL Divergence loss - grab this function from the quantization code
def loss_fn_kd(student_logits, teacher_logits, T):
    """
    Compute the knowledge-distillation (KD) loss given outputs from student and teacher
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """


    teacher_soft_logits = F.softmax(teacher_logits / T, dim=1)

    teacher_soft_logits = teacher_soft_logits.float()
    student_soft_logits = F.log_softmax(student_logits/T, dim=1)


    #For KL(p||q), p is the teacher distribution (the target distribution), and
    KD_loss = nn.KLDivLoss(reduction='batchmean')(student_soft_logits, teacher_soft_logits)
    KD_loss = (T ** 2) * KD_loss
    # KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)

    return KD_loss

def loss_fn_smooth_labels(model_logits, target_smooth_labels):

    model_probs = F.log_softmax(model_logits)
    smooth_label_loss = nn.KLDivLoss(reduction='batchmean')(model_probs, target_smooth_labels)

    return smooth_label_loss


# TODO: future work: the fim w.r.t the input BERT embeddings should be exactly calculable
#   This is because the last layer is either a linear regression layer or a logistic classification layer
def compute_FIM(bert_model, embeddings, predictions):
    # Take the classification layer of the Bert model as the embeddings and take
    pass

