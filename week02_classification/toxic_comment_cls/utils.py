import numpy as np
import pickle


def load_glove_model(file):
    print("Loading Glove Model")
    f = open(file,'r')
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    print(len(gloveModel)," words loaded!")
    return gloveModel

def initialise_word_embedding(file="/search/odin/nenggong/CRL_EGPG/pretrained/sgns.sogou.char"):
    vocab = pickle.load(open("./train/word2idx.pkl",'rb'))
    glove_emb = load_glove_model(file)
    word_emb = np.zeros((len(vocab),300))
    miss_num = 0
    for word,idx in vocab.items():
        if word in glove_emb:
            word_emb[idx] = glove_emb[word]
            continue
        miss_num+=1
    print(str(miss_num)+" words are not in the glove embedding")
    return word_emb