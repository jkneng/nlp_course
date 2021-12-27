import torch
from torch.utils.data import Dataset
import os
import pickle

class ToxicData(Dataset):
    def __init__(self, root, dcat, max_len) -> None:
        super(ToxicData, self).__init__()
        src_path = os.path.join(root, dcat + '/src.pkl')
        tgt_path = os.path.join(root, dcat + '/tgt.pkl')
        self.src = pickle.load(open(src_path, 'rb'))
        self.tgt = pickle.load(open(tgt_path, 'rb'))
        self.max_len = max_len
    
    def __getitem__(self, index):
        input = torch.tensor(self.src[index][:self.max_len])
        output = torch.tensor(self.tgt[index])
        return input, output
    
    def __len__(self):
        return len(self.src)

    