import os
import random
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchinfo import summary

class DubinsLSTMEncoderDecoder(nn.Module):
    def __init__(self, input_dim=3, cond_dim=8, hidden_dim=64, num_layers=4, output_dim=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2)
        )
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, conds, target_seq=None, lengths=None, teacher_forcing_ratio=0.0, 
                seq_len: int = 50):
        B = conds.size(0)
        device = conds.device
        cond_embed = self.decoder_fc(conds)
        cond_embed = cond_embed.unsqueeze(1)  # (B, 1, hidden_dim)
        if target_seq is not None:
            run_len = target_seq.size(1)
        else:
            run_len = seq_len
        # Initialize hidden and cell states from cond_embed
        h = cond_embed[:,:self.hidden_dim ].repeat(self.num_layers, 1, 1)  # (num_layers, B, hidden_dim)
        c = cond_embed[:,self.hidden_dim:].repeat(self.num_layers, 1, 1)  # (num_layers, B, hidden_dim)

        # h = torch.zeros(self.num_layers, B, self.hidden_dim, device=device)
        # c = torch.zeros(self.num_layers, B, self.hidden_dim, device=device)
        out_seq = []

        prev_out = torch.ones(B, 1, 3, device=device)*0.5
        prev_out[:, :, 2] = 0.0  # initial gamma=0.0
        for t in range(run_len):
            out, (h, c) = self.lstm(prev_out, (h, c))
            pred = self.fc_out(out)
            out_seq.append(pred)

            if target_seq is not None:
                if lengths is None:
                    use_tf = (random.random() < teacher_forcing_ratio)
                    prev_out = target_seq[:, t:t+1, :] if use_tf else pred
                else:
                    rand = torch.rand(B, device=device)
                    use_tf = (rand < teacher_forcing_ratio) & (t < lengths)
                    use_tf = use_tf.view(B, 1, 1)
                    prev_target = target_seq[:, t:t+1, :]
                    prev_out = torch.where(use_tf, prev_target, pred)
            else:
                prev_out = pred

        return torch.cat(out_seq, dim=1)
