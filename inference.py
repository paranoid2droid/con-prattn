"""
@Project: controllable-syllable-level-lyrics-generation-from-melody-with-prior-attention
@File: inference.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
"""
# This script performs inference by loading a trained melody-to-lyrics model,
# generating syllable-level lyrics for a chosen melody sequence, and producing
# a corresponding MIDI file with the generated lyrics.

import numpy as np

from utils import sequences_to_texts, remove_start_and_end
import os
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from transformer_multi_regularization_flag import TransformerGeneratorRegularizationFlag
from dataset_features_flags import LyricsMelodyDataset, LyricsMelodyDataModule
from hparams import *
from utils import create_midi_pattern_from_discretized_data_with_lyrics
import json
import random
import torch.nn.functional as F


# Set random seed for reproducibility
pl.seed_everything(42)

# Ensure deterministic behavior on GPU for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configure device: use GPU if available, else CPU
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

if __name__ == '__main__':
    # Define base folder for data and tokenizers
    data_folder = ''
    tokenizer_folder = os.path.join(data_folder, 'tokenizers')

    # Load dataset sequences for inference
    lyrics_melody_dataset = LyricsMelodyDataset(data=os.path.join(data_folder, 'sequences.npy'),
                                                tokenizer_syllable_path=os.path.join(tokenizer_folder, 'tokenizer_syllable.pkl'),
                                                tokenizer_word_path=os.path.join(tokenizer_folder, 'tokenizer_word.pkl'),
                                                tokenizer_pitch_path=os.path.join(tokenizer_folder, 'tokenizer_note.pkl'),
                                                tokenizer_duration_path=os.path.join(tokenizer_folder, 'tokenizer_duration.pkl'),
                                                tokenizer_rest_path=os.path.join(tokenizer_folder, 'tokenizer_rest.pkl')
                                                )

    # Initialize data module for inference splits
    lyrics_melody_data_module = LyricsMelodyDataModule(data_train=os.path.join(data_folder, 'sequences_train.npy'),
                                                       data_val=os.path.join(data_folder, 'sequences_valid.npy'),
                                                       data_test=os.path.join(data_folder, 'sequences_test.npy'),
                                                       tokenizer_syllable_path=os.path.join(tokenizer_folder,'tokenizer_syllable.pkl'),
                                                       tokenizer_word_path=os.path.join(tokenizer_folder,'tokenizer_word.pkl'),
                                                       tokenizer_pitch_path=os.path.join(tokenizer_folder,'tokenizer_note.pkl'),
                                                       tokenizer_duration_path=os.path.join(tokenizer_folder,'tokenizer_duration.pkl'),
                                                       tokenizer_rest_path=os.path.join(tokenizer_folder,'tokenizer_rest.pkl'),
                                                       batch_size=BATCH_SIZE,
                                                       num_workers=NUM_WORKERS)

    # Create DataLoaders for inference
    train_loader = lyrics_melody_data_module.train_dataloader()
    val_loader = lyrics_melody_data_module.val_dataloader()
    test_loader = lyrics_melody_data_module.test_dataloader()

    # Retrieve vocabulary sizes and special token indices
    vocab_size_syllable, vocab_size_word, vocab_size_pitch, vocab_size_duration, vocab_size_rest = lyrics_melody_dataset.get_vocab_size()
    padding_idx_syllable, padding_idx_word, padding_idx_pitch, padding_idx_duration, padding_idx_rest = lyrics_melody_dataset.get_padding_ids()
    start_id_syllable, start_id_word, start_id_pitch, start_id_duration, start_id_rest = lyrics_melody_dataset.get_start_ids()

    # Set up Trainer and load pretrained model checkpoint
    # Check whether pretrained model exists. If yes, load it and skip training
    # pretrained_gen_filename = os.path.join(CHECKPOINT_PATH, "test.ckpt")
    pretrained_filename = ''

    trainer_mle = pl.Trainer(gpus=1 if str(device).startswith("cuda") else 0)

    model = TransformerGeneratorRegularizationFlag.load_from_checkpoint(pretrained_filename)

    # Select a sample index from the test set for generation
    song_idx = 422

    # Prepare input melody, lyrics tokens, and flags for generation
    data_all = lyrics_melody_data_module.test_dataset.data
    data_lyrics = data_all[song_idx, :, 0]
    data_melody = data_all[song_idx, :, 2:5]
    flags = data_all[song_idx, :, 5]

    # Optionally override flags to specify custom prior attention patterns
    prattn = [0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 1, 3, 2, 0]
    flags[1:15] = prattn

    # Generate lyrics sequence using the model
    gen_lyrics = model.generate(data_melody, flags)

    # Convert ground-truth sequences to one-hot for decoding
    gt_lyrics_samples = F.one_hot(torch.LongTensor(data_lyrics[:-1]), vocab_size_syllable).float()
    melody_p_samples = F.one_hot(torch.LongTensor(data_melody[:-1, 0]), vocab_size_pitch).float()
    melody_d_samples = F.one_hot(torch.LongTensor(data_melody[:-1, 1]), vocab_size_duration).float()
    melody_r_samples = F.one_hot(torch.LongTensor(data_melody[:-1, 2]), vocab_size_rest).float()

    # Decode token sequences back to text (remove start/end tokens later)
    gt_lyrics = sequences_to_texts(gt_lyrics_samples, tokenizer_lyr_path=os.path.join(tokenizer_folder,'tokenizer_syllable.pkl'))
    melody_p = sequences_to_texts(melody_p_samples, tokenizer_lyr_path=os.path.join(tokenizer_folder,'tokenizer_note.pkl'))
    melody_d = sequences_to_texts(melody_d_samples, tokenizer_lyr_path=os.path.join(tokenizer_folder,'tokenizer_duration.pkl'))
    melody_r = sequences_to_texts(melody_r_samples, tokenizer_lyr_path=os.path.join(tokenizer_folder,'tokenizer_rest.pkl'))

    # Strip start and end tokens from decoded texts
    gt_lyrics = remove_start_and_end(gt_lyrics)
    melody_p = remove_start_and_end(melody_p)
    melody_d = remove_start_and_end(melody_d)
    melody_r = remove_start_and_end(melody_r)

    # Prepare generated lyrics and melody arrays for MIDI synthesis
    lyrics = gen_lyrics[0].split()
    melody = np.vstack((np.array(melody_p[0].split(), dtype=np.float32),
                        np.array(melody_d[0].split(), dtype=np.float32),
                        np.array(melody_r[0].split(), dtype=np.float32)))
    gt_lyrics = gt_lyrics[0].split()

    # Create a MIDI pattern with generated lyrics annotations
    gen_midi = create_midi_pattern_from_discretized_data_with_lyrics(melody.T, lyrics)

    # Save the generated MIDI file to disk
    file_name = 'idx422_transformer.mid'
    # file_name = 'idx422_prattn_' + str(prattn) + '.mid'
    gen_midi.write('./generated_midi/' + file_name)
    # print(f"Melody can be found at ../../subjective_evaluation/quality/{MIDI_NAME}.mid")
