from transformers.modeling_bert import *
from modelOutputDataclass import ASRErrorSequenceClassifierOutput
from transformers import BertTokenizer
from utils import *
from lossesRegularize import *
'''
The forward() function of this model allows the following two methods of regularization with soft labels
these can be applied at the same time
1. logits from a teacher model (regularize using distillation with a teacher model)
 


Follows the same structure as BertForSequenceClassification found here:
https://huggingface.co/transformers/_modules/transformers/modeling_bert.html#BertForSequenceClassification
'''

#This class is a modified version of the BertForSequenceClassification dataset found in transformers.modeling_bert
class ASRBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, alpha_distil=None, alpha_soft_label=None, temp=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.alpha_distil = alpha_distil
        self.alpha_soft_label = alpha_soft_label
        self.distil_T = temp
        self.init_weights()

    # @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     tokenizer_class=_TOKENIZER_FOR_DOC,
    #     checkpoint="bert-base-uncased",
    #     output_type=SequenceClassifierOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        teacher_logits=None,
        soft_labels=None, #Inclusion of soft labels
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # pdb.set_trace()
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        standardLoss = None
        totalLoss = None
        distil_loss = None
        soft_label_loss = None

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                standardLoss = loss_fct(logits.view(-1), labels.view(-1))
            else:   # baseline: no soft labels
                loss_fct = CrossEntropyLoss()
                standardLoss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            totalLoss = standardLoss

            if teacher_logits is not None:  # if soft labels are also passed
                distil_loss = loss_fn_kd(student_logits=logits, teacher_logits=teacher_logits, T=self.distil_T)
                totalLoss += self.alpha_distil * distil_loss

            if soft_labels is not None: # can ALSO use soft labels in addition to the
                soft_label_loss = loss_fn_smooth_labels(model_logits=logits, target_smooth_labels=soft_labels)
                totalLoss += self.alpha_soft_label * soft_label_loss

        if not return_dict:
            if distil_loss is not None:
                output = (totalLoss, standardLoss, distil_loss) + outputs[2:]
            else:
                output = (logits,) + outputs[2:]

            return ((totalLoss,) + output) if totalLoss is not None else output

        return ASRErrorSequenceClassifierOutput(
            loss=totalLoss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            distillation_loss=distil_loss,
            soft_label_loss=soft_label_loss
        )




