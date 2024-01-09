import sklearn
import numpy as np
import itertools
from torch.utils.data import Dataset
from transformers import BertTokenizer
from data_augmentation import Wordnet
from nltk.tokenize import word_tokenize

import random
import torch

class ABSADataset(Dataset):
    """
        Dataset class: Loads and transforms the dataset: Encoding of categories & polarity, Tokenization of the sentences
        Inherits pytorch dataset class and can be used by pytorch dataloader

        Data can be augmented using EDA technique: synonym replacement  and swap target words while taking into account the specifics of ABSA
        The logic of the data augmentation methods has been derived from the two following papers:
         -  "Data Augmentation in a Hybrid Approach for Aspect-Based Sentiment Analysis" by Tomas Liesting, Flavius Frasincar, Maria Mihaela Trusca
            https://arxiv.org/abs/2103.15912
         -  "Improving Short Text Classification Through Global Augmentation Methods" by Vukosi Marivate & Tshephisho Sefara 
            https://link.springer.com/chapter/10.1007/978-3-030-57321-8_21
    """
    def __init__(self, fname, tokenizer, augment=False, swap=False):
        with open(fname, 'r', encoding='UTF-8') as fin:
            lines = [line.strip().split("\t") for line in fin if line.strip()]

        #Encode aspect categories as numbers
        categories = [element[1] for element in lines]
        cat_encoder = sklearn.preprocessing.LabelEncoder()
        cat_encoder.fit(categories)
        le_cat = cat_encoder.transform(categories)

        self.tokenizer=tokenizer
        self.augment=augment
        self.swap=swap

        all_data = []
        for i, line in enumerate(lines):

            #Encode polarity as numbers
            if line[0] == "positive":
                polarity = 2
            elif line[0] == "negative":
                polarity = 0
            elif line[0] == "neutral":
                polarity = 1
            else:
                raise ValueError("Polarity problem")

            #Gather category and aspect
            category = le_cat[i]
            aspect = line[2]

            #Gather surrounding text right and left of keyword
            index_colon = line[3].index(':')
            start_num = int(line[3][:index_colon])
            end_num = int(line[3][index_colon+1:len(line[3])])
            text_left = line[4][:start_num].strip()
            text_right = line[4][end_num:].strip()

            # Augment the left and right text using wordnet synonyms replacement
            if self.augment:
                text_left= self.synonyms_replacement(text_left)
                text_right=self.synonyms_replacement(text_right)

            #Create features
            text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            context_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            left_indices = tokenizer.text_to_sequence(text_left)
            left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            right_with_aspect_indices = tokenizer.text_to_sequence(aspect + " " + text_right, reverse=True)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_len = np.sum(left_indices != 0)
            aspect_len = np.sum(aspect_indices != 0)
            aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)

            text_len = np.sum(text_indices != 0)
            concat_bert_indices = tokenizer.text_to_sequence(
                '[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
            concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
            concat_segments_indices = pad_and_truncate(concat_segments_indices, tokenizer.max_seq_len)

            text_bert_indices = tokenizer.text_to_sequence(
                "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
            aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

            #Get features
            data = {
                'concat_bert_indices': concat_bert_indices,
                'concat_segments_indices': concat_segments_indices,
                'text_bert_indices': text_bert_indices,
                'aspect_bert_indices': aspect_bert_indices,
                'text_indices': text_indices,
                'context_indices': context_indices,
                'left_indices': left_indices,
                'left_with_aspect_indices': left_with_aspect_indices,
                'right_indices': right_indices,
                'right_with_aspect_indices': right_with_aspect_indices,
                'aspect_indices': aspect_indices,
                'aspect_boundary': aspect_boundary,
                'polarity': polarity,
                'category': category
            }

            all_data.append(data)

        # Perform target swap across sentences of the same category
        if self.swap:
            all_data = self.target_swap_augmentation(all_data)
        
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def synonyms_replacement(self, text_left):
        # Perform synonyms replacement
        t = Wordnet(v=True ,n=True, p=0.5)
        new_left=t.augment(text_left)
        return new_left

    def target_swap_augmentation(self, data):
        # Group data by category
        category_data = {}
        for item in data:
            cat = item['category']
            if cat not in category_data:
                category_data[cat] = []
            category_data[cat].append(item)

        # Swap targets for each category
        augmented_data = []
        for cat, items in category_data.items():
            # Shuffle the list of sentences
            random.shuffle(items)
            # Iterate over the shuffled list of sentences, swapping each sentence with the next one in the list
            for i in range(0, len(items), 2):
                if i + 1 < len(items):
                    item_i, item_j = self.swap_targets(items[i], items[i+1])
                    augmented_data.extend([item_i, item_j])

        return augmented_data

    
    def replace_target(self,text, old_target, new_target):
        tokens = word_tokenize(text)
        old_target_tokens = word_tokenize(old_target)
        new_target_tokens = word_tokenize(new_target)

        for i, token in enumerate(tokens):
            if token == old_target_tokens[0] and tokens[i:i+len(old_target_tokens)] == old_target_tokens:
                tokens[i:i+len(old_target_tokens)] = new_target_tokens
                break

        return ' '.join(tokens)



    def swap_targets(self, item_i, item_j):
        # Get target and text for each item
        target_i = ' '.join(self.tokenizer.bert_tokenizer.convert_ids_to_tokens(item_i['aspect_bert_indices'], skip_special_tokens=True))
        target_j = ' '.join(self.tokenizer.bert_tokenizer.convert_ids_to_tokens(item_j['aspect_bert_indices'], skip_special_tokens=True))
        text_i = ' '.join(self.tokenizer.bert_tokenizer.convert_ids_to_tokens(item_i['text_bert_indices'], skip_special_tokens=True))
        text_j = ' '.join(self.tokenizer.bert_tokenizer.convert_ids_to_tokens(item_j['text_bert_indices'], skip_special_tokens=True))

        # Swap targets and update texts
        text_i = self.replace_target(text_i, target_i, target_j)
        text_j = self.replace_target(text_j, target_j, target_i)

        # Update items with new texts
        item_i['text_bert_indices'] = self.tokenizer.text_to_sequence("[CLS] " + text_i + " [SEP]")
        item_j['text_bert_indices'] = self.tokenizer.text_to_sequence("[CLS] " + text_j + " [SEP]")

        return item_i, item_j

class Tokenizer():
    '''
    Tokenizer class using existing BERT-Tokenizer
    '''
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)

def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x

