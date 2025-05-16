"""
@Project: controllable-syllable-level-lyrics-generation-from-melody-with-prior-attention
@File: dataset_features_flags.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
"""

# This module defines the LyricsMelodyDataset and LyricsMelodyDataModule
# for loading and batching paired melody and syllable-level lyrics data.

from torch.utils.data import Dataset, DataLoader
import torch
from utils import load_tokenizer
import pytorch_lightning as pl
import numpy as np


# Dataset class for paired melody-to-lyrics data at the syllable level.
class LyricsMelodyDataset(Dataset):

    # Initialize dataset: load numpy data and tokenizers for syllable, word, pitch, duration, and rest.
    def __init__(self, data, tokenizer_syllable_path, tokenizer_word_path,
                 tokenizer_pitch_path, tokenizer_duration_path, tokenizer_rest_path):
        self.data = np.load(data)
        self.tokenizer_syllable = load_tokenizer(tokenizer_syllable_path)
        self.tokenizer_word = load_tokenizer(tokenizer_word_path)
        self.tokenizer_pitch = load_tokenizer(tokenizer_pitch_path)
        self.tokenizer_duration = load_tokenizer(tokenizer_duration_path)
        self.tokenizer_rest = load_tokenizer(tokenizer_rest_path)

    # Return the total number of samples in the dataset.
    def __len__(self):
        return len(self.data)

    # Retrieve a single sample: returns input melody attributes, target lyrics tokens, and flags as tensors.
    def __getitem__(self, idx):
        syllable = self.data[idx, :, 0]
        word = self.data[idx, :, 1]
        pitch = self.data[idx, :, 2]
        duration = self.data[idx, :, 3]
        rest = self.data[idx, :, 4]
        flag = self.data[idx, :, 5]

        return (torch.tensor(pitch, dtype=torch.long), torch.tensor(duration, dtype=torch.long), torch.tensor(rest, dtype=torch.long)), \
                   (torch.tensor(syllable, dtype=torch.long), torch.tensor(word, dtype=torch.long)), torch.tensor(flag, dtype=torch.long)

    # Compute and return vocabulary sizes for all tokenizers (syllable, word, pitch, duration, rest).
    def get_vocab_size(self):
        vocab_size_syllable = len(self.tokenizer_syllable.word_index) + 1
        vocab_size_word = len(self.tokenizer_word.word_index) + 1
        vocab_size_pitch = len(self.tokenizer_pitch.word_index) + 1
        vocab_size_duration = len(self.tokenizer_duration.word_index) + 1
        vocab_size_rest = len(self.tokenizer_rest.word_index) + 1

        return vocab_size_syllable, vocab_size_word, vocab_size_pitch, vocab_size_duration, vocab_size_rest

    # Return the padding token index (<eos>) for each tokenizer.
    def get_padding_ids(self, padding_id="<eos>"):
        padding_id_syllable = self.tokenizer_syllable.word_index[padding_id]
        padding_id_word = self.tokenizer_word.word_index[padding_id]
        padding_id_pitch = self.tokenizer_pitch.word_index[padding_id]
        padding_id_duration = self.tokenizer_duration.word_index[padding_id]
        padding_id_rest = self.tokenizer_rest.word_index[padding_id]

        return padding_id_syllable, padding_id_word, padding_id_pitch, padding_id_duration, padding_id_rest

    # Return the start token index (<bos>) for each tokenizer.
    def get_start_ids(self, start_id="<bos>"):
        start_id_syllable = self.tokenizer_syllable.word_index[start_id]
        start_id_word = self.tokenizer_word.word_index[start_id]
        start_id_pitch = self.tokenizer_pitch.word_index[start_id]
        start_id_duration = self.tokenizer_duration.word_index[start_id]
        start_id_rest = self.tokenizer_rest.word_index[start_id]

        return start_id_syllable, start_id_word, start_id_pitch, start_id_duration, start_id_rest


# LightningDataModule to provide train/validation/test DataLoaders for the LyricsMelodyDataset.
class LyricsMelodyDataModule(pl.LightningDataModule):

    # Initialize DataModule: set paths for train/val/test data, tokenizers, batch size, and number of workers.
    def __init__(self, data_train, data_val, data_test,
                 tokenizer_syllable_path, tokenizer_word_path,
                 tokenizer_pitch_path, tokenizer_duration_path, tokenizer_rest_path,
                 batch_size: int, num_workers: int = 0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = LyricsMelodyDataset(data=data_train,
                                                 tokenizer_syllable_path=tokenizer_syllable_path,
                                                 tokenizer_word_path=tokenizer_word_path,
                                                 tokenizer_pitch_path=tokenizer_pitch_path,
                                                 tokenizer_duration_path=tokenizer_duration_path,
                                                 tokenizer_rest_path=tokenizer_rest_path)
        self.val_dataset = LyricsMelodyDataset(data=data_val,
                                               tokenizer_syllable_path=tokenizer_syllable_path,
                                               tokenizer_word_path=tokenizer_word_path,
                                               tokenizer_pitch_path=tokenizer_pitch_path,
                                               tokenizer_duration_path=tokenizer_duration_path,
                                               tokenizer_rest_path=tokenizer_rest_path)
        self.test_dataset = LyricsMelodyDataset(data=data_test,
                                                tokenizer_syllable_path=tokenizer_syllable_path,
                                                tokenizer_word_path=tokenizer_word_path,
                                                tokenizer_pitch_path=tokenizer_pitch_path,
                                                tokenizer_duration_path=tokenizer_duration_path,
                                                tokenizer_rest_path=tokenizer_rest_path)

    # Create and return the DataLoader for the training split.
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    # Create and return the DataLoader for the validation split.
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    # Create and return the DataLoader for the test split.
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
