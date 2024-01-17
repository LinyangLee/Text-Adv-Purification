
import random
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertForSequenceClassification, BertForMaskedLM, BertModel, AutoConfig, AutoTokenizer, BertPreTrainedModel


class BertTriTask(BertForMaskedLM):
    def __init__(self, config):
        super(BertTriTask, self).__init__(config)
        self.mlm_probability = 0.15
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.emotion_cls = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        class_labels=None,
        output_labels=None,
        emotion_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        token_scores = self.cls(sequence_output)  # B, seq-len, vocab
        cls_scores = self.classifier(pooled_output)
        emotion_scores = self.emotion_cls(sequence_output)
        loss_fct = CrossEntropyLoss(ignore_index=-100)  # -100 index = padding token

        loss = 0
        if output_labels is not None:
            loss += loss_fct(token_scores.view(-1, self.config.vocab_size), output_labels.view(-1))
        if class_labels is not None:
            loss += loss_fct(cls_scores.view(-1, self.num_labels), class_labels.view(-1))
        if emotion_labels is not None:
            loss += loss_fct(emotion_scores.view(-1, self.num_labels), emotion_labels.view(-1))

        return loss, cls_scores, token_scores, emotion_scores, class_labels, output_labels, emotion_labels


class RobustBert(BertForMaskedLM):
    def __init__(self, config):
        super(RobustBert, self).__init__(config)

        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

        # self.mlm = BertForMaskedLM.from_pretrained(mlm_model_path)
        # self.fine_tune_classifier = BertForSequenceClassification.from_pretrained(tgt_model_path, config=config)
        # self.mlm.eval()
        # self.fine_tune_classifier.eval()

    def forward_this_model(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def forward_masked_lm(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        token_score = self.cls(sequence_output)
        return torch.argmax(token_score, dim=-1), token_score

    def forward_inference_mass(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        total_expand = 32
        with torch.no_grad():
            input_ids_expand = input_ids.repeat(total_expand, 1)  # N, L
            mask_expand = attention_mask.repeat(total_expand, 1)  # N, L
            seg_expand = token_type_ids.repeat(total_expand, 1)  # N, L

            probability_matrix = torch.full(input_ids_expand.shape, 0.15, device=input_ids.device)
            masked_indices = torch.bernoulli(probability_matrix).long()
            masked_indices = masked_indices * 103  # make mask
            masked_indices_nt = masked_indices.eq(0)
            masked_ids = input_ids_expand * masked_indices_nt + masked_indices
            pre_pos = 0
            result_logits = None
            for i in range(int(total_expand/8)):
                cur_pos = (i + 1) * 8
                cur_input_ids = masked_ids[pre_pos: cur_pos]
                cur_attention_mask = mask_expand[pre_pos: cur_pos]
                cur_token_type_ids = seg_expand[pre_pos: cur_pos]
                pre_pos = cur_pos
                rebuild_ids, _ = self.forward_masked_lm(cur_input_ids, cur_attention_mask, cur_token_type_ids)
                logits = self.forward_this_model(rebuild_ids, cur_attention_mask, cur_token_type_ids)
                # logits = torch.mean(logits, dim=0, keepdim=True)
                result_logits = torch.cat([result_logits, logits], dim=0) if result_logits is not None else logits

            return (torch.mean(result_logits, dim=0, keepdim=True),)
            # rebuild_ids, _ = self.forward_masked_lm(masked_ids, mask_expand, seg_expand)
            # logits = self.forward_this_model(rebuild_ids, mask_expand, seg_expand)
            # logits = torch.mean(logits, dim=0, keepdim=True)
            # return (logits,)

    def forward_inference_insert(self, input_ids, attention_mask=None, token_type_ids=None):
        total_expand = 16
        seq_length = torch.sum(attention_mask, -1)

        with torch.no_grad():
            input_ids_expand_list = []  # N, L
            mask_expand_list = []  # N, L
            seg_expand_list = []  # N, L

            for i in range(total_expand):
                insert_index = random.randint(1, seq_length)
                if insert_index == 512:
                    insert_index = insert_index - 1
                tmp_input_ids = input_ids.clone()
                input_ids_expand_list.append(torch.cat([tmp_input_ids[:insert_index], torch.tensor([103]).to(device="cuda"), tmp_input_ids[insert_index: -1]]).view(1, -1))
                tmp_attention_mask = attention_mask.clone()
                mask_expand_list.append(torch.cat([tmp_attention_mask[:insert_index], tmp_attention_mask[insert_index-1: -1]]).view(1, -1))
                tmp_token_type = token_type_ids.clone()
                seg_expand_list.append(torch.cat([tmp_token_type[: insert_index], tmp_token_type[insert_index-1: -1]]).view(1, -1))
            for i in range(total_expand):
                insert_index = random.randint(1, seq_length)
                if insert_index == 512:
                    insert_index = insert_index - 1
                tmp_input_ids = input_ids.clone()
                input_ids_expand_list.append(torch.cat(
                    [tmp_input_ids[:insert_index], tmp_input_ids[insert_index+1:], tmp_input_ids[-1:]]).view(1, -1))
                tmp_attention_mask = attention_mask.clone()
                mask_expand_list.append(
                    torch.cat([tmp_attention_mask[:insert_index], tmp_attention_mask[insert_index+1:], tmp_attention_mask[-1:]]).view(1, -1))
                tmp_token_type = token_type_ids.clone()
                seg_expand_list.append(
                    torch.cat([tmp_token_type[: insert_index], tmp_token_type[insert_index+1:], tmp_token_type[-1:]]).view(1, -1))
            inputs_ids_expand = torch.cat(input_ids_expand_list, dim=0)
            mask_expand = torch.cat(mask_expand_list, dim=0)
            seg_expand = torch.cat(seg_expand_list, dim=0)

            pre_pos = 0
            result_logits = None
            for i in range(int(total_expand / 8)):
                cur_pos = (i + 1) * 8
                cur_input_ids = inputs_ids_expand[pre_pos: cur_pos]
                cur_attention_mask = mask_expand[pre_pos: cur_pos]
                cur_token_type_ids = seg_expand[pre_pos: cur_pos]
                pre_pos = cur_pos
                logits = self.forward_this_model(cur_input_ids, cur_attention_mask, cur_token_type_ids)
                result_logits = torch.cat([result_logits, logits], dim=0) if result_logits is not None else logits
            return (torch.mean(result_logits, dim=0, keepdim=True),)


    def forward_inference_entropy(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        with torch.no_grad():
            _, token_score = self.forward_masked_lm(input_ids, attention_mask, token_type_ids)
            # token-score B seq-len, vocab-size
            token_prob = torch.softmax(token_score, dim=-1)
            entropy_ = torch.mean((- token_prob * torch.log(token_prob)), dim=-1)  # B seq-len
            seq_len = input_ids.size(1)
            _, top_k_masks = torch.topk(entropy_, k=int(0.15 * seq_len), dim=-1, largest=True)  # B, k

            masked_ids = input_ids.clone()

            masked_ids[torch.arange(attention_mask.size(0)).unsqueeze(1), top_k_masks] = 103  # fill in with [MASK]

            rebuild_ids, _ = self.forward_masked_lm(masked_ids, attention_mask, token_type_ids)

            logits = self.forward_this_model(rebuild_ids, attention_mask, token_type_ids)

        return (logits,)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                class_labels=None,
                output_labels=None,
                emotion_labels=None,
                ):
        batch_size = input_ids.shape[0]
        result = None

        for seq in range(batch_size):
            with torch.no_grad():
                logits_0 = self.forward_inference_mass(input_ids[seq, :], attention_mask=attention_mask[seq, :], token_type_ids=token_type_ids[seq, :])[0]
                logits_1 = self.forward_inference_insert(input_ids[seq, :], attention_mask=attention_mask[seq, :], token_type_ids=token_type_ids[seq, :])[0]
                logits = (logits_0 + logits_1) / 2
            result = logits if result is None else torch.cat([result, logits], dim=0)
        return (result,)

    # def to(self, device):
    #     self.mlm.to(device)
    #     self.fine_tune_classifier.to(device)



import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertForSequenceClassification, BertForMaskedLM, BertModel


class BertTriTask(BertForMaskedLM):
    def __init__(self, config):
        super(BertTriTask, self).__init__(config)
        self.mlm_probability = 0.15
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.emotion_cls = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        class_labels=None,
        output_labels=None,
        emotion_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        token_scores = self.cls(sequence_output)  # B, seq-len, vocab
        cls_scores = self.classifier(pooled_output)
        emotion_scores = self.emotion_cls(sequence_output)
        loss_fct = CrossEntropyLoss(ignore_index=-100)  # -100 index = padding token

        loss = 0
        if output_labels is not None:
            loss += loss_fct(token_scores.view(-1, self.config.vocab_size), output_labels.view(-1))
        if class_labels is not None:
            loss += loss_fct(cls_scores.view(-1, self.num_labels), class_labels.view(-1))
        if emotion_labels is not None:
            loss += loss_fct(emotion_scores.view(-1, self.num_labels), emotion_labels.view(-1))

        return loss, cls_scores, token_scores, emotion_scores, class_labels, output_labels, emotion_labels


class RobustBert_test(object):
    def __init__(self, config, tgt_model_path=None, mlm_model_path=None):
        # super().__init__(config)

        self.num_labels = config.num_labels

        self.mlm = BertForMaskedLM.from_pretrained(mlm_model_path)
        self.fine_tune_classifier = BertForSequenceClassification.from_pretrained(tgt_model_path, config=config)
        self.mlm.eval()
        self.fine_tune_classifier.eval()

    def __call__(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        return self.forward(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)

    def forward_this_model(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.fine_tune_classifier.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs[1]

        pooled_output = self.fine_tune_classifier.dropout(pooled_output)
        logits = self.fine_tune_classifier.classifier(pooled_output)
        return logits

    def forward_masked_lm(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.mlm.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        token_score = self.mlm.cls(sequence_output)
        return torch.argmax(token_score, dim=-1), token_score

    def forward_inference_mass(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        total_expand = 32
        with torch.no_grad():
            input_ids_expand = input_ids.repeat(total_expand, 1)  # N, L
            mask_expand = attention_mask.repeat(total_expand, 1)  # N, L
            seg_expand = token_type_ids.repeat(total_expand, 1)  # N, L

            probability_matrix = torch.full(input_ids_expand.shape, 0.15, device=input_ids.device)
            masked_indices = torch.bernoulli(probability_matrix).long()
            masked_indices = masked_indices * 103  # make mask
            masked_indices_nt = masked_indices.eq(0)
            masked_ids = input_ids_expand * masked_indices_nt + masked_indices

            rebuild_ids, _ = self.forward_masked_lm(masked_ids, mask_expand, seg_expand)
            logits = self.forward_this_model(rebuild_ids, mask_expand, seg_expand)
            logits = torch.mean(logits, dim=0, keepdim=True)
            return (logits,)

    def forward_inference_entropy(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        with torch.no_grad():
            _, token_score = self.forward_masked_lm(input_ids, attention_mask, token_type_ids)
            # token-score B seq-len, vocab-size
            token_prob = torch.softmax(token_score, dim=-1)
            entropy_ = torch.mean((- token_prob * torch.log(token_prob)), dim=-1)  # B seq-len
            seq_len = input_ids.size(1)
            _, top_k_masks = torch.topk(entropy_, k=int(0.15 * seq_len), dim=-1, largest=True)  # B, k

            masked_ids = input_ids.clone()

            masked_ids[torch.arange(attention_mask.size(0)).unsqueeze(1), top_k_masks] = 103  # fill in with [MASK]

            rebuild_ids, _ = self.forward_masked_lm(masked_ids, attention_mask, token_type_ids)

            logits = self.forward_this_model(rebuild_ids, attention_mask, token_type_ids)

        return (logits,)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        with torch.no_grad():
            logits_0 = self.forward_inference_mass(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels,
                                                   )[0]
            # logits_1 = self.forward_inference_entropy(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels,
            #                                      )[0]
            # logits = (logits_0 + logits_1) / 2
            logits = logits_0

        return (logits,)

    def to(self, device):
        self.mlm.to(device)
        self.fine_tune_classifier.to(device)
