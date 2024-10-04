#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 14:53:11 2023

@author: ahannan
"""

import io
import torch
import editdistance
from conf import *
import torchaudio.transforms as T
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        # 30
        ^ 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        ' 29
        $ 31
        @ 0
        """
        # ^=<SOS> 1
        # $=<EOS> 31
        # #=<PAD> 30
        # @=<blank> for ctc
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[28] = ' '
    def text_to_int(self,text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = 28#self.char_map['']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence
    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i.detach().item()])
        return ''.join(string)#.replace('', ' ')


def pad_sequence(batch, padvalue):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=padvalue)
    return batch.permute(0, 2, 1)

text_transform=TextTransform()

################################################################################
################################################################################

spec_transform = T.Spectrogram(n_fft=n_fft * 2, \
                               hop_length=hop_length, \
                               win_length=win_length)

melspec_transform = T.MelScale(sample_rate=sample_rate, n_mels=n_mels, n_stft=n_fft+1)

################################################################################
################################################################################
