import torch
import torch.nn as nn

from mdense import MDense

class Model(nn.module):
    def __init__(self,
            nb_mels,
            rnn_units1,
            rnn_units2,
            embed_size,
            frame_size,
            pcm_bits,
            output_size,
            training=False):

        padding = 0 if training else 1
        self.fconv1 = nn.Sequential(
            nn.Conv1d(nb_mels, 128, 3, padding=padding),
            nn.Tanh()
        )
        self.fconv2 = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=padding),
            nn.Tanh()
        )
        self.fdense1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh()
        )
        self.fdense2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh()
        )
        self.embed = nn.Embedding(256, embed_size)
        self.embed_size = embed_size
        self.frame_size = frame_size
        self.rnn_units1 = rnn_units1
        self.rnn_units2 = rnn_units2
    def forward(self, pcm, feat):
        cfeat = self.fconv1(feat)
        cfeat = self.fconv2(cfeat)
        cfeat = self.fdense1(cfeat)
        cfeat = self.fdense2(cfeat)
        cfeat = cfeat.repeat(self.frame_size, 1)
        cpcm = self.embed(pcm).reshape(-1, embed_size)
        rnn_in1 = torch.cat([cpcm, cfeat], dim=-1)
        gru_out1, _ = nn.GRU(rnn_in1.shape[-1], self.rnn_units1, batch_firse=False)
        rnn_in2 = torch.cat([gru_out1, cfeat], dim=-1)
        gru_out2, _ = nn.GRU(rnn_in2.shape[-1], self.rnn_units2, batch_firse=False)
        ulaw_prob = MDense(gru_out2)
        return ulaw_prob



