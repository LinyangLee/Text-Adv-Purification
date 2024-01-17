

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertForSequenceClassification, BertForMaskedLM, BertModel, BertTokenizer
import random


def l2norm(X):
    norm = torch.pow(X, 2).sum(dim=-1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


class BertAT(BertForMaskedLM):
    def __init__(self, config):
        super(BertAT, self).__init__(config)
        self.mlm_probability = 0.15
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def get_masked(self, input_ids, mlm_probability=0.15, label=None):
        output_labels = input_ids.clone()
        probability_matrix = torch.full(input_ids.shape, mlm_probability)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        output_labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(output_labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = 103  # hard code mask index

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(
            torch.full(output_labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # hard code random word
        random_words = torch.randint(30522, output_labels.shape, dtype=torch.long, device=output_labels.device)
        input_ids[indices_random] = random_words[indices_random]

        return input_ids, output_labels

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            class_labels=None,
            **kwargs
    ):

        # ensemble masks

        tr_loss = 0
        adv_steps = 2
        adv_init_mag = 1e-1
        adv_max_norm = 2e-1
        adv_lr = 1e-2

        # should do the mask outside the adv train process
        masked_ids, output_labels = self.get_masked(input_ids)

        embeds_init = self.bert.embeddings.word_embeddings(masked_ids)

        delta = torch.zeros_like(embeds_init).uniform_(-1, 1)
        dims = torch.tensor(768, device=delta.device).float()
        mag = adv_init_mag / torch.sqrt(dims)  # B
        bs, seq_len = input_ids.size()
        delta = delta * mag.view(-1, 1, 1)

        for astep in range(adv_steps):

            delta.requires_grad_()
            inputs_embeds = embeds_init + delta
            # use input embeds

            outputs = self.bert(
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds
            )
            sequence_output = outputs[0]
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)

            token_scores = self.cls(sequence_output)  # B, seq-len, vocab
            cls_scores = self.classifier(pooled_output)

            # -100 index = padding token  no padding
            loss_fct = CrossEntropyLoss()

            loss = 0
            if output_labels is not None:
                loss += loss_fct(token_scores.view(-1, self.config.vocab_size), output_labels.view(-1))
            if class_labels is not None:
                loss += loss_fct(cls_scores.view(-1, self.num_labels), class_labels.view(-1))

            tr_loss += loss
            loss.backward()

            if astep == adv_steps - 1:
                break

            # get grad on delta
            delta_grad = delta.grad.clone().detach()

            # clip

            # grad-norm
            denorm = torch.norm(delta_grad, dim=-1).view(bs, seq_len, 1)  # B seq-len 1
            denorm = torch.clamp(denorm, min=1e-8)
            # add the delta with grads
            delta = (delta + adv_lr * delta_grad / denorm).detach()  # B seq-len D

            # normalize new delta at token-level
            delta_norm = torch.norm(delta, p=2, dim=-1).detach()  # B seq-len
            mean_norm, _ = torch.max(delta_norm, dim=-1, keepdim=True)  # B,1
            # reweight-delta using scaling
            reweights_tok = (delta_norm / mean_norm).view(bs, seq_len, 1)  # B seq-len, 1
            delta = delta * reweights_tok

            # reweight the exceed delta
            delta_norm = torch.norm(delta.view(bs, -1).float(), p=2, dim=1).detach()
            exceed_mask = (delta_norm > adv_max_norm).to(embeds_init)
            reweights = (adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)  # B 1 1

            # detach delta and embeds init for next iteration
            delta = (delta * reweights).detach()
            embeds_init = self.bert.embeddings.word_embeddings(masked_ids)

        return tr_loss

    def forward_inference(self,
                          input_ids=None,
                          attention_mask=None,
                          token_type_ids=None,
                          inputs_embeds=None,
                          class_labels=None,
                          ):
        masked_ids = input_ids
        output_labels = input_ids.clone()

        outputs = self.bert(
            masked_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        token_scores = self.cls(sequence_output)  # B, seq-len, vocab
        cls_scores = self.classifier(pooled_output)

        loss_fct = CrossEntropyLoss(ignore_index=-100)  # -100 index = padding token

        loss = 0
        if output_labels is not None:
            loss += loss_fct(token_scores.view(-1, self.config.vocab_size), output_labels.view(-1))
        if class_labels is not None:
            loss += loss_fct(cls_scores.view(-1, self.num_labels), class_labels.view(-1))

        return loss, cls_scores, token_scores, class_labels, output_labels


class RobustBert(object):
    def __init__(self, config, tgt_model_path=None, mlm_model_path=None, tokenizer: BertTokenizer = None):
        # super().__init__(config)

        self.num_labels = config.num_labels

        self.mlm = BertAT.from_pretrained(mlm_model_path)
        self.fine_tune_classifier = BertAT.from_pretrained(tgt_model_path, config=config)
        self.mlm.eval()
        self.fine_tune_classifier.eval()
        self.tokenizer = BertTokenizer.from_pretrained(mlm_model_path)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.fine_tune_classifier.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs[1]
        seq_output = outputs[0]
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

    def forward_inference_whole_word_mask(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        total_expand = 16
        text_list = input_ids.cpu().numpy().tolist()[-1]
        sentence = self.tokenizer.decode(text_list, skip_special_tokens=True)
        words = self.tokenizer.basic_tokenizer.tokenize(sentence)
        seq_length = len(words)
        mask_size = int(seq_length * 0.15)
        sample_index = [i for i in range(seq_length)]
        mask_token = self.tokenizer.mask_token
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        for _ in range(total_expand):
            mask_index = random.sample(sample_index, mask_size)
            tmp_words = words.copy()
            for index in mask_index:
                sub_word = self.tokenizer(tmp_words[index], add_special_tokens=False)["input_ids"]
                mask_length = len(sub_word)
                mask_words = " ".join([mask_token for _ in range(mask_length)])
                tmp_words[index] = mask_words
            tokenizer_result = self.tokenizer.encode_plus(" ".join(tmp_words), None, add_special_tokens=True, max_length=512, truncation=True)
            input_ids_list.append(tokenizer_result["input_ids"])
            attention_mask_list.append(tokenizer_result["attention_mask"])
            token_type_ids_list.append(tokenizer_result["token_type_ids"])

        with torch.no_grad():
            input_ids_expand = torch.tensor(input_ids_list, dtype=torch.long).to(device='cuda')
            mask_expand = torch.tensor(attention_mask_list, dtype=torch.long).to(device='cuda')
            seg_expand = torch.tensor(token_type_ids_list, dtype=torch.long).to(device='cuda')

            rebuild_ids, _ = self.forward_masked_lm(input_ids_expand, mask_expand, seg_expand)
            logits = self.forward(rebuild_ids, mask_expand, seg_expand)  # N, num-labels
            logits = l2norm(logits)

            logits = torch.mean(logits, dim=0, keepdim=True)

            return (logits,)

    def forward_inference_mass(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        total_expand = 8
        with torch.no_grad():
            input_ids_expand = input_ids.repeat(total_expand, 1)  # N, L
            mask_expand = attention_mask.repeat(total_expand, 1)  # N, L
            seg_expand = token_type_ids.repeat(total_expand, 1)  # N, L

            probability_matrix = torch.full(input_ids_expand.shape, 0.20, device=input_ids.device)
            masked_indices = torch.bernoulli(probability_matrix).long()
            masked_indices = masked_indices * 103  # make mask
            masked_indices_nt = masked_indices.eq(0)
            masked_ids = input_ids_expand * masked_indices_nt + masked_indices

            rebuild_ids, _ = self.forward_masked_lm(masked_ids, mask_expand, seg_expand)
            logits = self.forward(rebuild_ids, mask_expand, seg_expand)  # N, num-labels
            logits = l2norm(logits)

            logits = torch.mean(logits, dim=0, keepdim=True)

            return (logits,)

    def forward_inference_mass_shift(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # do insert and delete
        total_expand = 8
        seq_length = int(torch.sum(attention_mask, -1))
        input_ids_expand_list = []  # N, L
        input_ids_delete_list = []
        mask_expand_list = []  # N, L
        mask_delete_list = []  # N, L
        seg_expand_list = []  # N, L
        seg_delete_list = []  # N, L

        with torch.no_grad():

            # do insert
            insert_token = 103
            for i in range(total_expand):
                input_ids_list = input_ids.squeeze().cpu().tolist()
                attention_mask_list = attention_mask.squeeze().cpu().tolist()
                token_type_ids_list = token_type_ids.squeeze().cpu().tolist()

                permute_place = random.sample(range(len(input_ids_list)), int(0.20 * len(input_ids_list)))

                inserted_input_ids = []
                deleted_input_ids = []
                inserted_mask = []
                deleted_mask = []
                inserted_segs = []
                deleted_segs = []

                for index, token in enumerate(input_ids_list):
                    inserted_input_ids.append(input_ids_list[index])
                    inserted_mask.append(attention_mask_list[index])
                    inserted_segs.append(token_type_ids_list[index])
                    if index in permute_place:
                        inserted_input_ids.append(insert_token)
                        inserted_mask.append(1)
                        inserted_segs.append(0)
                    else:
                        deleted_input_ids.append(input_ids_list[index])
                        deleted_mask.append(attention_mask_list[index])
                        deleted_segs.append(token_type_ids_list[index])

                # tensorize insert ids
                inserted_input_ids = torch.tensor(inserted_input_ids[:seq_length]).unsqueeze(0)
                inserted_mask = torch.tensor(inserted_mask[:seq_length]).unsqueeze(0)
                inserted_segs = torch.tensor(inserted_segs[:seq_length]).unsqueeze(0)
                input_ids_expand_list.append(inserted_input_ids)
                mask_expand_list.append(inserted_mask)
                seg_expand_list.append(inserted_segs)

                # pad and tensorize delete ids
                pad_length = seq_length - len(deleted_input_ids)
                deleted_input_ids = deleted_input_ids + [0 for _ in range(pad_length)]
                deleted_mask = deleted_mask + [0 for _ in range(pad_length)]
                deleted_segs = deleted_segs + [0 for _ in range(pad_length)]
                deleted_input_ids = torch.tensor(deleted_input_ids).unsqueeze(0)
                deleted_mask = torch.tensor(deleted_mask).unsqueeze(0)
                deleted_segs = torch.tensor(deleted_segs).unsqueeze(0)
                input_ids_delete_list.append(deleted_input_ids)
                mask_delete_list.append(deleted_mask)
                seg_delete_list.append(deleted_segs)

            input_ids_expand_list = torch.cat(input_ids_expand_list, dim=0).to('cuda')
            mask_expand_list = torch.cat(mask_expand_list, dim=0).to('cuda')
            seg_expand_list = torch.cat(seg_expand_list, dim=0).to('cuda')

            input_ids_delete_list = torch.cat(input_ids_delete_list, dim=0).to('cuda')
            mask_delete_list = torch.cat(mask_delete_list, dim=0).to('cuda')
            seg_delete_list = torch.cat(seg_delete_list, dim=0).to('cuda')

            # rebuild the inserted masks
            # inserted seqs logits
            rebuild_ids, _ = self.forward_masked_lm(input_ids_expand_list, mask_expand_list, seg_expand_list)
            logits = self.forward(rebuild_ids, mask_expand_list, seg_expand_list)
            logits = l2norm(logits)

            logits_insert_mean = torch.mean(logits, dim=0, keepdim=True)

            # deleted seqs logits
            # logits = self.forward(input_ids_delete_list, mask_delete_list, seg_delete_list)
            # logits_delete_mean = torch.mean(logits, dim=0, keepdim=True)

            # logits = (logits_insert_mean + logits_delete_mean) / 2

            logits = logits_insert_mean

            return (logits,)

    def forward_inference(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        with torch.no_grad():
            logits_0 = \
                self.forward_inference_mass(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                            labels=labels,
                                            )[0]
            logits_1 = \
                self.forward_inference_mass_shift(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                                  labels=labels,
                                                  )[0]
            logits = (logits_0 + logits_1) / 2
            # logits = self.forward(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            # logits = logits_0

        return (logits,)

    def to(self, device):
        self.mlm.to(device)
        self.fine_tune_classifier.to(device)
