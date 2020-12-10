import torch
import torch.nn as nn

from mdense import MDense
from loss import sample_from_discretized_mix_logistic

class Model(nn.module):
    def __init__(self,
            nb_mels,
            rnn_units1,
            rnn_units2,
            embed_size,
            frame_size,
            num_mixture,
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
        self.output_size = num_mixture * 3
        self.gru_a = nn.GRU(384, self.rnn_units1, batch_firse=False)
        self.gru_b = nn.GRU(self.rnn_units1 + 128, self.rnn_units2, batch_firse=False)
        self.md = Mdense(self.rnn_units2, output_size)
 
    def forward(self, pcm, feat):
        cfeat = self.fconv1(feat)
        cfeat = self.fconv2(cfeat)
        cfeat = self.fdense1(cfeat)
        cfeat = self.fdense2(cfeat)
        cfeat = cfeat.repeat(self.frame_size, 1)
        cpcm = self.embed(pcm).reshape(-1, embed_size)
        rnn_in1 = torch.cat([cpcm, cfeat], dim=-1)
        gru_out1, _ = self.gru_a(rnn_in1)
        rnn_in2 = torch.cat([gru_out1, cfeat], dim=-1)
        gru_out2, _ = self.gru_b(rnn_in2)
        ulaw_prob = self.md(gru_out2)
        return ulaw_prob

    def encoder(self, feat):
        cfeat = self.fconv1(feat)
        cfeat = self.fconv2(cfeat)
        cfeat = self.fdense1(cfeat)
        cfeat = self.fdense2(cfeat)
        return cfeat
 
    def decoder(self, pcm, dec_feat, dec_state1, dec_state2):
        cpcm = self.embed(pcm).reshape(-1, embed_size)
        dec_rnn_in = torch.cat([cpcm, dec_feat], dim=-1)
        dec_gru_out1, state1 = self.gru_a(dec_rnn_in, dec_state1)
        dec_gru_out2, state2 = self.gru_b(torch.cat([dec_gru_out1, dec_feat], dim=-1), dec_state2)
        dec_ulaw_prob = self.md(dec_gru_out2)
        sample_func = sample_from_discretized_mix_logistic(dec_ulaw_prob)
        sample = sample_func(dec_ulaw_prob)
        return sample, state1, state2


    def sparse_gru_a(self, global_step, t_start, t_end, interval, density):
        final_density = density
        if global_step < t_start or ((global_step - t_start) % interval != 0 and global_step < t_end):
            break
        else:
            p = self.gru_a.weight_hh_l0.numpy()
            nb = p.shape[1]//p.shape[0]
            N = p.shape[0]
            for k in range(nb):
                density = final_density[k]
                if global_step < t_end:
                    r = 1 - (global_step - t_start) / (t_end - t_start)
                    density = 1 - (1 - final_density[k] ) * (1 - r*r*r) 
                A = p[:, k*N:(k+1)*N]
                A = A - np.diag(np.diag(A))
                A = np.transpose(A, (1, 0))
                L=np.reshape(A, (N, N//16, 16))
                S=np.sum(L*L, axis=-1)
                SS=np.sort(np.reshape(S, (-1,)))
                thresh = SS[round(N*N//16*(1-density))]
                mask = (S>=thresh).astype('float32')
                mask = np.repeat(mask, 16, axis=1)
                mask = np.minimum(1, mask + np.diag(np.ones((N,))))
                mask = np.transpose(mask, (1, 0))
                p[:, k*N:(k+1)*N] = p[:, k*N:(k+1)*N]*mask
            
            self.gru_a.weight_hh_l0.data.copy_(torch.Tensor(p))    

