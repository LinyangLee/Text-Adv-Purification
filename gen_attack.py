

import warnings
import os

import torch
import torch.nn as nn
import ujson as json
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import BertConfig, BertTokenizer, AlbertTokenizer
from transformers import BertForSequenceClassification, BertForMaskedLM
import copy
import argparse
import numpy as np
from model import RobustBert

filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves']
filter_words = set(filter_words)


def get_sim_embed(embed_path, sim_path):
    id2word = {}
    word2id = {}

    with open(embed_path, 'r', encoding='utf-8') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in id2word:
                id2word[len(id2word)] = word
                word2id[word] = len(id2word) - 1

    cos_sim = np.load(sim_path)
    return cos_sim, word2id, id2word


def get_data_cls(data_path):
    features = []
    lines = open(data_path, 'r', encoding='utf-8').readlines()[:]
    if data_path.endswith(".json"):
        for (i, line) in enumerate(lines):
            line = json.loads(line.strip())
            seq = line["sentence"]
            label = int(line["label"])
            features.append([seq, label])
    elif data_path.endswith(".csv") or data_path.endswith(".tsv") or data_path.endswith(".txt"):
        for i, line in enumerate(lines):
            split = line.strip('\n').split('\t')
            label = int(split[-1])
            seq = split[0]
            features.append([seq, label])
    else:
        raise NotImplementedError("Data format illegal, please check")

    return features


class Feature(object):
    def __init__(self, seq_a, label):
        self.label = label
        self.seq = seq_a
        self.final_adverse = seq_a
        self.query = 0
        self.change = 0
        self.success = ''
        self.sim = 0.0
        self.changes = []


def _tokenize(seq, tokenizer):
    seq = seq.replace('\n', '').lower()
    words = seq.split(' ')

    sub_words = []
    keys = []
    index = 0
    for word in words:
        sub = tokenizer.tokenize(word)
        sub_words += sub
        keys.append([index, index + len(sub)])
        index += len(sub)

    return words, sub_words, keys


def _get_masked(words):
    len_text = len(words)
    masked_words = []
    for i in range(len_text - 1):
        masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
    # list of words
    return masked_words


def get_important_scores(words, tgt_model, orig_prob, orig_label, orig_probs, tokenizer, batch_size, max_length):
    masked_words = _get_masked(words)
    texts = [' '.join(words) for words in masked_words]  # list of text of masked words
    all_input_ids = []
    all_masks = []
    all_segs = []
    for text in texts:
        inputs = tokenizer.encode_plus(text, None, add_special_tokens=True, max_length=max_length, truncation=True)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + (padding_length * [0])
        token_type_ids = token_type_ids + (padding_length * [0])
        attention_mask = attention_mask + (padding_length * [0])
        all_input_ids.append(input_ids)
        all_masks.append(attention_mask)
        all_segs.append(token_type_ids)
    seqs = torch.tensor(all_input_ids, dtype=torch.long)
    masks = torch.tensor(all_masks, dtype=torch.long)
    segs = torch.tensor(all_segs, dtype=torch.long)
    seqs = seqs.to('cuda')
    masks = masks.to('cuda')
    segs = segs.to('cuda')

    # iterative

    leave_1_probs = []
    for masked_input, mask_, segs_ in zip(seqs, masks, segs):
        leave_1_logit = tgt_model.forward_inference(masked_input.unsqueeze(0), mask_.unsqueeze(0), segs_.unsqueeze(0),
                                                    torch.tensor(int(orig_label)).unsqueeze(0).to('cuda'))[0]
        # B num-label  cut-off the gradients ? here ?
        leave_1_probs.append(leave_1_logit.detach())

    leave_1_probs = torch.cat(leave_1_probs, dim=0)  # words, num-label
    leave_1_probs = torch.softmax(leave_1_probs, -1)  #
    leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
    import_scores = (orig_prob
                     - leave_1_probs[:, orig_label]
                     +
                     (leave_1_probs_argmax != orig_label).float()
                     * (leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))
                     ).data.cpu().numpy()

    return import_scores


def get_substitutes_mlm(substitutes, tokenizer):
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len != 1:
        return words

    else:
        for (i) in (substitutes[0]):
            words.append(tokenizer._convert_id_to_token(int(i)))
        return words


def get_substitutes_text_fooler(tgt_word, w2i, i2w, cos_mat, k):
    if tgt_word not in w2i:
        return []
    word_idx = w2i[tgt_word]
    cos_sims = cos_mat[word_idx]  # list of sims
    cos_sims = torch.tensor(cos_sims)
    _, topk_idxs = torch.topk(cos_sims, k=k, dim=-1)
    output_words = [i2w[int(topk_idx)] for topk_idx in topk_idxs]
    return output_words


def attack(feature, tgt_model, mlm_model, mlm_tokenizer, tgt_tokenizer, k, batch_size, max_length=512, cos_mat=None, w2i={}, i2w={},
           attack_type='mlm'):
    # MLM-process

    words, sub_words, keys = _tokenize(feature.seq, mlm_tokenizer)

    # original label
    inputs = tgt_tokenizer.encode_plus(feature.seq, None, add_special_tokens=True, max_length=max_length, truncation=True)
    input_ids, attention_mask, token_type_ids = torch.tensor(inputs["input_ids"]), \
                                                torch.tensor(inputs["attention_mask"]), \
                                                torch.tensor(inputs["token_type_ids"])

    orig_probs = tgt_model.forward_inference(input_ids.unsqueeze(0).to('cuda'),
                                             attention_mask.unsqueeze(0).to('cuda'),
                                             token_type_ids.unsqueeze(0).to('cuda'),
                                             torch.tensor(feature.label).unsqueeze(0).to('cuda'),
                                             )[0].squeeze()
    orig_probs = torch.softmax(orig_probs, -1)
    orig_label = torch.argmax(orig_probs)
    current_prob = orig_probs.max()

    if orig_label != feature.label:
        feature.success = 'direct-success'
        return feature

    if attack_type == 'mlm':
        sub_words = ['[CLS]'] + sub_words[:max_length - 2] + ['[SEP]']
        input_ids_ = torch.tensor([mlm_tokenizer.convert_tokens_to_ids(sub_words)])
        word_predictions = mlm_model(input_ids_.to('cuda'))[0].squeeze()  # seq-len(sub) vocab
        word_pred_scores_all, word_predictions = torch.topk(word_predictions, k, -1)  # seq-len k

        word_predictions = word_predictions[1:len(sub_words) + 1, :]
        word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]

    important_scores = get_important_scores(words, tgt_model, current_prob, orig_label, orig_probs,
                                            mlm_tokenizer, batch_size, max_length)
    feature.query += int(len(words))
    list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)
    final_words = copy.deepcopy(words)
    masked_words = copy.deepcopy(words)

    for idx, top_index in enumerate(list_of_index):
        if feature.change > int(0.4 * (len(words))):
            feature.success = 'exceed'  # exceed
            return feature

        tgt_word = words[top_index[0]]

        if tgt_word in filter_words:
            continue
        if keys[top_index[0]][0] > max_length - 2:
            continue

        if attack_type == 'mlm':
            # bert-attack method
            substitutes = word_predictions[keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
            word_pred_scores = word_pred_scores_all[keys[top_index[0]][0]:keys[top_index[0]][1]]
            substitutes = get_substitutes_mlm(substitutes, mlm_tokenizer)
        if attack_type == 'textfooler':
            # text-fooler method
            substitutes = get_substitutes_text_fooler(tgt_word, w2i, i2w, cos_mat, k)

        most_gap = 0.0
        candidate = None

        for substitute in substitutes:

            if substitute == tgt_word or '##' in substitute or substitute in filter_words:
                continue  # filter out original word , sub-word

            if attack_type == 'textfooler' or attack_type == 'mlm':
                if substitute in w2i and tgt_word in w2i:
                    if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.50:
                        continue

            temp_replace = final_words
            temp_replace[top_index[0]] = substitute
            temp_text = mlm_tokenizer.convert_tokens_to_string(temp_replace)
            inputs = tgt_tokenizer.encode_plus(temp_text, None, add_special_tokens=True, max_length=max_length, truncation=True)
            input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).to('cuda')
            mask_ = torch.tensor(inputs["attention_mask"]).unsqueeze(0).to('cuda')
            seg_ = torch.tensor(inputs["token_type_ids"]).unsqueeze(0).to('cuda')

            temp_prob = tgt_model.forward_inference(input_ids, mask_, seg_, torch.tensor(feature.label).unsqueeze(0).to('cuda'),
                                                    )[0].squeeze()

            feature.query += 1
            temp_prob = torch.softmax(temp_prob, -1)
            temp_label = torch.argmax(temp_prob)

            if temp_label != orig_label:
                feature.change += 1
                final_words[top_index[0]] = substitute
                feature.changes.append([keys[top_index[0]][0], substitute, tgt_word])
                feature.changes.append([0, str(int(temp_label)), str(int(orig_label))])
                feature.final_adverse = temp_text
                feature.success = 'success'
                return feature
            else:

                label_prob = temp_prob[orig_label]
                gap = current_prob - label_prob
                if gap > most_gap:
                    most_gap = gap
                    candidate = substitute

        if most_gap > 0:
            feature.change += 1
            feature.changes.append([keys[top_index[0]][0], candidate, tgt_word])
            current_prob = current_prob - most_gap
            final_words[top_index[0]] = candidate
        # else:
        #     final_words[top_index[0]] = tgt_word

    feature.final_adverse = (mlm_tokenizer.convert_tokens_to_string(final_words))
    feature.success = 'failed'
    return feature


def evaluate(features):
    acc = 0
    origin_success = 0
    total = 0
    total_q = 0
    total_change = 0
    total_word = 0
    for feat in features:
        if 'success' in feat.success:

            acc += 1
            total_q += feat.query
            total_change += feat.change
            total_word += len(feat.seq.split(' '))

            if 'direct' in feat.success:
                origin_success += 1

        total += 1

    if acc == 0:
        acc = 0.1
        total_word = 1

    suc = float(acc / total)

    query = float(total_q / acc)
    change_rate = float(total_change / total_word)

    origin_acc = 1 - origin_success / total
    after_atk = 1 - suc

    print('acc/aft-atk-acc {:.6f}/ {:.6f}, query-num {:.4f}, change-rate {:.4f}'.format(origin_acc, after_atk, query, change_rate))


def dump_features(features, output):
    outputs = []

    for feature in features:
        outputs.append({'label': feature.label,
                        'success': feature.success,
                        'change': feature.change,
                        'num_word': len(feature.seq.split(' ')),
                        'query': feature.query,
                        'changes': feature.changes,
                        'seq_a': feature.seq,
                        'adv': feature.final_adverse,
                        })
    output_json = output
    json.dump(outputs, open(output_json, 'w'), indent=2, ensure_ascii=False)

    print('finished dump')


def run_attack():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="./data/xxx")
    parser.add_argument("--mlm_path", type=str, help="xxx mlm")
    parser.add_argument("--tgt_path", type=str, help="xxx classifier")

    parser.add_argument("--output_dir", type=str, help="train file")
    parser.add_argument("--use_sim_mat", type=int, help='whether use cosine_similarity to filter out atonyms')

    parser.add_argument("--attack_type", type=str, default='mlm', help='')

    parser.add_argument("--sim_embed", type=str, help='')
    parser.add_argument("--cosine_npy", type=str, help='')

    parser.add_argument("--start", type=int, help="start step, for multi-thread process")
    parser.add_argument("--end", type=int, help="end step, for multi-thread process")
    parser.add_argument("--num_label", type=int, )
    parser.add_argument("--k", type=int, )

    args = parser.parse_args()

    print('start process')

    tokenizer_mlm = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenizer_tgt = BertTokenizer.from_pretrained(args.tgt_path, do_lower_case=True)

    config_atk = BertConfig.from_pretrained(args.mlm_path)
    mlm_model = BertForMaskedLM.from_pretrained(args.mlm_path, config=config_atk)
    mlm_model.to('cuda')

    config_tgt = BertConfig.from_pretrained(args.tgt_path, num_labels=args.num_label)
    if True:
        tgt_model = RobustBert(config_tgt, args.tgt_path, args.tgt_path)
    else:
        tgt_model = BertForSequenceClassification.from_pretrained(args.tgt_path, config=config_tgt)

    tgt_model.to('cuda')
    features = get_data_cls(args.data_path)

    if args.use_sim_mat == 1:
        cos_mat, w2i, i2w = get_sim_embed(args.sim_embed, args.cosine_npy)
    else:
        cos_mat, w2i, i2w = None, {}, {}

    features_output = []

    temp_suc = 0
    temp_fail = 0
    for index, feature in enumerate(features[args.start:args.end]):
        seq_a, label = feature
        feat = Feature(seq_a, label)
        print('\r number {:d} '.format(index), end='')
        # print(feat.seq[:100], feat.label)

        feat = attack(feat, tgt_model, mlm_model, tokenizer_tgt, tokenizer_mlm, args.k, batch_size=1, max_length=512,
                      cos_mat=cos_mat, w2i=w2i, i2w=i2w, attack_type=args.attack_type)

        # print(feat.changes, feat.change, feat.query, feat.success)
        if 'success' in feat.success:
            temp_suc += 1
            print('suc', temp_suc, '/', temp_fail, end='')
        else:
            temp_fail += 1
            print('fail', temp_suc, '/', temp_fail, end='')
        features_output.append(feat)

    evaluate(features_output)

    dump_features(features_output, args.output_dir)


if __name__ == '__main__':
    run_attack()
