"""
@Project: controllable-syllable-level-lyrics-generation-from-melody-with-prior-attention
@File: utils.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
"""

import os
import pickle
import torch
import numpy as np
import torch.optim as optim
import pretty_midi
import json


def load_tokenizer(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
        tokenizer = data['tokenizer']
    return tokenizer


def remove_start_and_end(lyrics):
    clean_lyrics = []
    for lyric in lyrics:
        removal_list = ["<bos>", "<eos>", "<BOS>", "<EOS>"]
        lyric_list = lyric.split()
        final_list = [word for word in lyric_list if word not in removal_list]
        final_string = ' '.join(final_list)
        clean_lyrics.append(final_string)

    return clean_lyrics


def sequences_to_texts(samples, tokenizer_lyr_path=None):
    tokenizer_lyr = load_tokenizer(tokenizer_lyr_path)

    if samples.ndim == 2:
        samples = samples.unsqueeze(0)

    sequences = torch.argmax(samples, dim=2)
    texts = tokenizer_lyr.sequences_to_texts(sequences.cpu().numpy())

    return texts


def get_fixed_temperature(temper, i, N, adapt):
    """A function to set up different temperature control policies"""

    if adapt == 'no':
        temper_var_np = 1.0  # no increase, origin: temper
    elif adapt == 'lin':
        temper_var_np = 1 + i / (N - 1) * (temper - 1)  # linear increase
    elif adapt == 'exp':
        temper_var_np = temper ** (i / N)  # exponential increase
    elif adapt == 'log':
        temper_var_np = 1 + (temper - 1) / np.log(N) * np.log(i + 1)  # logarithm increase
    elif adapt == 'sigmoid':
        temper_var_np = (temper - 1) * 1 / (1 + np.exp((N / 2 - i) * 20 / N)) + 1  # sigmoid increase
    elif adapt == 'quad':
        temper_var_np = (temper - 1) / (N - 1) ** 2 * i ** 2 + 1
    elif adapt == 'sqrt':
        temper_var_np = (temper - 1) / np.sqrt(N - 1) * np.sqrt(i) + 1
    else:
        raise Exception("Unknown adapt type!")

    return temper_var_np


def truncated_normal_(tensor, mean=0, std=1):
    """
    Implemented by @ruotianluo
    See https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

def create_midi_pattern_from_discretized_data(discretized_sample):
    new_midi = pretty_midi.PrettyMIDI()
    voice = pretty_midi.Instrument(1)  # It's here to change the used instruments !
    tempo = 120
    ActualTime = 0  # Time since the beginning of the song, in seconds
    for i in range(0,len(discretized_sample)):
        length = discretized_sample[i][1] * 60 / tempo  # Conversion Duration to Time
        if i < len(discretized_sample) - 1:
            gap = discretized_sample[i + 1][2] * 60 / tempo
        else:
            gap = 0  # The Last element doesn't have a gap
        note = pretty_midi.Note(velocity=100, pitch=int(discretized_sample[i][0]), start=ActualTime,
                                end=ActualTime + length)
        voice.notes.append(note)
        ActualTime += length + gap  # Update of the time

    new_midi.instruments.append(voice)

    return new_midi

def create_midi_pattern_from_discretized_data_with_lyrics(discretized_sample, syllable_list):

    assert len(discretized_sample) == len(syllable_list)

    new_midi = pretty_midi.PrettyMIDI()
    voice = pretty_midi.Instrument(1)  # It's here to change the used instruments !
    tempo = 120
    ActualTime = 0  # Time since the beginning of the song, in seconds
    for i in range(0,len(discretized_sample)):
        length = discretized_sample[i][1] * 60 / tempo  # Conversion Duration to Time
        if i < len(discretized_sample) - 1:
            gap = discretized_sample[i + 1][2] * 60 / tempo
        else:
            gap = 0  # The Last element doesn't have a gap
        note = pretty_midi.Note(velocity=100, pitch=int(discretized_sample[i][0]), start=ActualTime,
                                end=ActualTime + length)
        voice.notes.append(note)

        lyric = pretty_midi.Lyric(text=syllable_list[i], time=ActualTime)
        new_midi.lyrics.append(lyric)

        ActualTime += length + gap  # Update of the time

    new_midi.instruments.append(voice)

    return new_midi

def load_lyrics_from_json(lyrics_file):
    with open(lyrics_file, 'r') as f:
        data = f.read()
    lyrics_syllable_list = json.loads(data)
    return lyrics_syllable_list
