import re

from torch._C import device
from model import TextCNN, TextLSTM
from dataset import ToxicData
from torch.utils.data import DataLoader
import argparse
import json
import torch.nn as nn
import torch.optim as optim
import torch
import os
from torch.nn.utils.rnn import pad_sequence
import sys
sys.path.append('.')
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_option():
     parser = argparse.ArgumentParser('argument for training')
     parser.add_argument('--model_conf', type=str, default='cnn.json', help='model config file')
     parser.add_argument('--batch_size', type=int, default=8, help='batch size')
     parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
     parser.add_argument('--epochs', type=int, default=25, help='epochs')
     parser.add_argument('--save_step_freq', type=int, default=200, help='save_step_freq')
     parser.add_argument('--model_path', type=str, default='saved_model/cnn_emb_init_bn', help='model_path')
     
     opt = parser.parse_args()
     return opt

def collate_fn(data):
    x, y = zip(*data)
    x_len = [len(d) for d in x]
    x = pad_sequence(x, batch_first=True)
    y = torch.tensor(y)
    return x, y, x_len

if __name__ == '__main__':
    opt = parse_option()
    model_cfg = json.load(open(opt.model_conf, 'r'))
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)
    
    root = './'
    max_len = 100
    train_data = ToxicData(root, 'train', max_len)
    train_dl = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)

    eval_data = ToxicData(root, 'eval', max_len)
    eval_dl = DataLoader(eval_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)

    # model = TextCNN(model_cfg)
    model = TextLSTM(model_cfg).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    from torch.utils.tensorboard import SummaryWriter
    writer_subfilename = opt.model_path.split('/')[1]
    writer = SummaryWriter(f'./tmp/train_log_{writer_subfilename}')

    steps = 0
    for epoch in range(opt.epochs):
        # training
        model.train()
        total_acc, total_cnt = 0.0, 0.0

        for i, (src_b, tgt_b, src_len) in enumerate(train_dl):
            src_b = src_b.to(device)
            tgt_b = tgt_b.to(device)
            # src_len = src_len.to(device)
            # pred_b = model(src_b)
            pred_b = model(src_b, src_len)
            # print(pred_b, pred_b.shape)
            # print(tgt_b, tgt_b.shape)
            train_loss_b = loss_fn(pred_b, tgt_b)
            total_acc += (pred_b.argmax(1) == tgt_b).sum().item()
            total_cnt += tgt_b.size(0)

            # backpropagation
            optimizer.zero_grad()
            train_loss_b.backward() # backpropagation and compute gradients
            optimizer.step() # parameters update
            
            steps += opt.batch_size * (i + 1)
            if i > 0 and steps % opt.save_step_freq == 0:
                torch.save(model.state_dict(), os.path.join(opt.model_path, f'model_{steps}.pth'))
                writer.add_scalar('train/loss', train_loss_b.cpu().item(), steps)
                writer.add_scalar('train/acc', total_acc/total_cnt, steps)
                total_total_acc, total_cnt = 0.0, 0.0
        # evaluation every epoch
        total_acc, total_cnt = 0.0, 0.0
        eloss = 0.0
        model.eval()
        with torch.no_grad():
            for src_b, tgt_b, src_len in eval_dl:
                pred_b = model(src_b, src_len)
                eloss += loss_fn(pred_b, tgt_b).item()
                total_acc += (pred_b.argmax(1) == tgt_b).sum().item()
                total_cnt += tgt_b.size(0)
        writer.add_scalar('eval/loss', eloss / len(eval_dl), epoch)
        writer.add_scalar('eval/acc', total_acc/total_cnt, epoch)
        print(f'eval loss {eloss / len(eval_dl)}, acc: {total_acc/total_cnt}')

    # prediction
    saved_model = TextLSTM(model_cfg).to(device)
    saved_model.load_state_dict(torch.load('saved_model/lstm/model_992400.pth'))
    for d in zip(eval_data.src, eval_data.tgt):
        pass
