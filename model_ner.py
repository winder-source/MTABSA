import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.models.bert.modeling_bert import BertForTokenClassification, BertPooler


class BERT_NER(BertForTokenClassification):

    def __init__(self, bert_base_model):
        super(BERT_NER, self).__init__(config=bert_base_model.config)
        self.bert4ner = bert_base_model

    def forward(self, input_ids,
                token_type_ids=None,
                attention_mask=None,
                valid_ids=None,
                label_mask=None,
                labels=None
                ):

        outputs = self.bert4ner(input_ids, token_type_ids, attention_mask)
        sequence_output = outputs[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device='cuda')
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        # 取所有有效字的均值
        valid_feature = valid_output.mean(dim=1)
        sequence_output = self.dropout(valid_output)
        ate_logits = self.classifier(sequence_output)

        return valid_feature, ate_logits, sequence_output
        # return outputs[1], ate_logits, sequence_output

        # if labels is not None:
        #     criterion_ate = CrossEntropyLoss(ignore_index=0)
        #     loss_ate = criterion_ate(ate_logits.view(-1, self.num_labels), labels.view(-1))
        #     return loss_ate
        # else:
        #     return ate_logits
