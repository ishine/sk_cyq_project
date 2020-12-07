import numpy as np 
import os
import torch

from hparams import hparams
from torch.utils.data import Dataset
from utils.ulaw import ulaw2lin, lin2ulaw
from tqdm import tqdm

class lpcnetDataset(Dataset):
    """
        处理数据
    """
    def __init__(self, args):
        self.file_list = args.file_list
        self.mel_dir = args.mel_dir
        self.pcm_dir = args.pcm_dir
        self.in_datas, self.mels, self.outputs = self.collect_data(self.file_list, self.mel_dir, self.pcm_dir)

    def __getitem__(self, index):
        in_data =  self.in_datas[index]
        mel = self.mels[index]
        output = self.outputs[index]

    def __len__(self):
        return len(self.outputs)

    def collect_data(self, file_list, mel_dir, pcm_dir):
        nb_mels = hparams.nb_mels
        feature_chunk_size = hparams.feature_chunk_size
        pcm_chunk_size = hparams.frame_size * hparams.feature_chunk_size
        
        in_datas, mels, out_excs = [], [], []
        with open(file_list) as fin:
            fids = [line.strip() for line in fin.readlines()]
        
        for fid in tqdm(fids):
            if hparams.model == 'MELNET':
                pcm = np.fromfile(os.path.join(pcm_dir, fid + '.s16'), dtype='int16')
                nb_frames = len(pcm) // (pcm_chunk_size)
                out_pcm = lin2ulaw(pcm[: nb_frames * pcm_chunk_size])
                in_pcm = np.concatenate([out_pcm[: 1], out_pcm[: -1]], axis=-1)
                out_exc = np.reshape(out_pcm, (nb_frames, pcm_chunk_size, 1))
                in_data = np.reshape(in_pcm, (nb_frames, pcm_chunk_size, 1))
                in_datas.append(in_data)
            else:
                pcm = np.fromfile(os.path.join(pcm_dir, fid + '.u8'), dtype='uint8')
                nb_frames = len(pcm) // (pcm_chunk_size * 4)
                pcm = pcm[: nb_frames * pcm_chunk_size * 4]
                sig = np.reshape(pcm[0:: 4], (nb_frames, pcm_chunk_size, 1))
                pred = np.reshape(pcm[1:: 4], (nb_frames, pcm_chunk_size, 1))
                in_exc = np.reshape(pcm[2:: 4], (nb_frames, pcm_chunk_size, 1))
                out_exc = np.reshape(pcm[3:: 4], (nb_frames, pcm_chunk_size, 1))
                in_data = np.concatenate([sig, pred, in_exc], axis=-1)
                in_datas.append(in_data)

            out_exc = out_exc / (2**8 - 1) * 2 - 1.0
            out_excs.append(out_exc)
            
            mel = np.fromfile(os.path.join(mel_dir, fid + '.mel'), dtype='float32')
            if mel.shape[0] % nb_mels != 0:
                print(fid)
            mel = mel[: nb_frames * feature_chunk_size * nb_mels]
            mel = np.reshape(mel, (nb_frames * feature_chunk_size, nb_mels))
            mel = np.reshape(mel, (nb_frames, feature_chunk_size, nb_mels))
            mels.append(mel)

        in_datas = np.concatenate(in_datas, axis=0)
        mels = np.concatenate(mels, axis=0)
        out_excs = np.concatenate(out_excs, axis=0)
        print("ulaw std = ", np.std(out_excs))

        fpad1 = np.concatenate([mels[0: 1, 0: 2, :], mels[: -1, -2:, :]], axis=0)
        fpad2 = np.concatenate([mels[1:, : 2, :], mels[0: 1, -2:, :]], axis=0)
        mels = np.concatenate([fpad1, mels, fpad2], axis=1)
        return in_datas, mels, out_excs
