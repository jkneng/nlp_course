from collections import Counter
import numpy
from torchtext.data.utils import get_tokenizer
from torch.utils.data import random_split
from torchtext.vocab import build_vocab_from_iterator

import pandas
import pickle

data = pandas.read_csv('../comments.tsv', sep='\t', header=0)

tokenizer = get_tokenizer('basic_english')
# print(tokenizer(data['comment_text'][0]))
word2idx, idx2word = {'PAD': 0}, {}

counter = Counter()
for c in data['comment_text']:
    counter.update(tokenizer(c))
idx = 1
for k, v in counter.items():
    word2idx[k] = idx
    idx2word[idx] = k
    idx += 1
print(len(word2idx))
# pickle.dump(word2idx, open('./train/word2idx.pkl', 'wb'))
# pickle.dump(idx2word, open('./train/idx2word.pkl', 'wb'))

# comment_lens = [len(d) for d in data['comment_text']]
# print(numpy.mean(comment_lens))
# print(numpy.std(comment_lens))
max_len = 300
train_size = int(len(data) * 0.8)
eval_size = len(data) - train_size
train_dataset, eval_dataset = random_split(list(zip(data['should_ban'], data['comment_text'])), [train_size, eval_size])

train_src_ids, train_tgt = [], []
for label, comment_text in train_dataset:
    src_ids = [word2idx[t] for t in tokenizer(comment_text)]
    train_src_ids.append(src_ids)

    train_tgt.append(int(label))
pickle.dump(train_src_ids, open('./train/src.pkl', 'wb'))
pickle.dump(train_tgt, open('./train/tgt.pkl', 'wb'))

eval_src_ids, eval_tgt = [], []
for label, comment_text in eval_dataset:
    src_ids = [word2idx[t] for t in tokenizer(comment_text)]
    eval_src_ids.append(src_ids)
    eval_tgt.append(int(label))
pickle.dump(eval_src_ids, open('./eval/src.pkl', 'wb'))
pickle.dump(eval_tgt, open('./eval/tgt.pkl', 'wb'))
