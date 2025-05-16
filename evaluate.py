"""
@Project: controllable-syllable-level-lyrics-generation-from-melody-with-prior-attention
@File: evaluate.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
"""

# This script evaluates a trained melody-to-lyrics model by generating lyrics
# on the test set and computing evaluation metrics such as ROUGE, BLEU, and BERT scores.

import numpy as np

from utils import sequences_to_texts, remove_start_and_end
import os
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from transformer_multi_regularization_flag import TransformerGeneratorRegularizationFlag
from dataset_features_flags import LyricsMelodyDataset, LyricsMelodyDataModule
from hparams import *
from evaluation import *
import json
import random


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

    # Load dataset sequences for generation and evaluation
    lyrics_melody_dataset = LyricsMelodyDataset(data=os.path.join(data_folder, 'sequences.npy'),
                                                tokenizer_syllable_path=os.path.join(tokenizer_folder, 'tokenizer_syllable.pkl'),
                                                tokenizer_word_path=os.path.join(tokenizer_folder, 'tokenizer_word.pkl'),
                                                tokenizer_pitch_path=os.path.join(tokenizer_folder, 'tokenizer_note.pkl'),
                                                tokenizer_duration_path=os.path.join(tokenizer_folder, 'tokenizer_duration.pkl'),
                                                tokenizer_rest_path=os.path.join(tokenizer_folder, 'tokenizer_rest.pkl'),
                                                )

    # Initialize data module for train/val/test splits
    lyrics_melody_data_module = LyricsMelodyDataModule(data_train=os.path.join(data_folder, 'sequences_train.npy'),
                                                       data_val=os.path.join(data_folder, 'sequences_valid.npy'),
                                                       data_test=os.path.join(data_folder, 'sequences_test.npy'),
                                                       tokenizer_syllable_path=os.path.join(tokenizer_folder,'tokenizer_syllable.pkl'),
                                                       tokenizer_word_path=os.path.join(tokenizer_folder,'tokenizer_word.pkl'),
                                                       tokenizer_pitch_path=os.path.join(tokenizer_folder,'tokenizer_note.pkl'),
                                                       tokenizer_duration_path=os.path.join(tokenizer_folder,'tokenizer_duration.pkl'),
                                                       tokenizer_rest_path=os.path.join(tokenizer_folder,'tokenizer_rest.pkl'),
                                                       batch_size=BATCH_SIZE,
                                                       num_workers=NUM_WORKERS,
                                                       )

    # Create PyTorch DataLoaders for evaluation
    train_loader = lyrics_melody_data_module.train_dataloader()
    val_loader = lyrics_melody_data_module.val_dataloader()
    test_loader = lyrics_melody_data_module.test_dataloader()

    # Retrieve vocabulary sizes and special token indices
    vocab_size_syllable, vocab_size_word, vocab_size_pitch, vocab_size_duration, vocab_size_rest = lyrics_melody_dataset.get_vocab_size()
    padding_idx_syllable, padding_idx_word, padding_idx_pitch, padding_idx_duration, padding_idx_rest = lyrics_melody_dataset.get_padding_ids()
    start_id_syllable, start_id_word, start_id_pitch, start_id_duration, start_id_rest = lyrics_melody_dataset.get_start_ids()

    # Check whether pretrained model exists. If yes, load it and skip training
    # pretrained_gen_filename = os.path.join(CHECKPOINT_PATH, "test.ckpt")
    pretrained_filename = ''

    # Initialize PyTorch Lightning Trainer for prediction
    trainer = pl.Trainer(gpus=1 if str(device).startswith("cuda") else 0)

    model = TransformerGeneratorRegularizationFlag.load_from_checkpoint(pretrained_filename)

    # Run prediction on the test set to generate lyrics and collect ground truth
    predictions = trainer.predict(model, dataloaders=test_loader)

    gen_lyrics = []
    gt_lyrics = []

    for batch in predictions:
        gen_lyrics = gen_lyrics + batch[0]
        gt_lyrics = gt_lyrics + batch[1]

    dir_str = pretrained_filename.split('/')[1]
    lyrics_dir = os.path.join('./lyrics', dir_str)

    os.makedirs(lyrics_dir, exist_ok=True)

    # Save generated and ground truth lyrics to JSON files
    with open(os.path.join(lyrics_dir, 'gen_lyrics.json'), 'w') as fp:
        json.dump(gen_lyrics, fp)

    with open(os.path.join(lyrics_dir, 'gt_lyrics.json'), 'w') as fp:
        json.dump(gt_lyrics, fp)

    # Compute ROUGE scores between generated and ground truth lyrics
    rouge_1, rouge_2, rouge_l = get_rouge_scores(gt_lyrics, gen_lyrics)
    print('Rouge 1/2/l (f1, precision, recall): ', rouge_1, rouge_2, rouge_l)

    # Compute BLEU scores between generated and ground truth lyrics
    bleu_2, bleu_3, bleu_4 = get_bleu_scores(gt_lyrics, gen_lyrics)
    print('Sentence BLEU 2/3/4: ', bleu_2, bleu_3, bleu_4)

    # Compute ChrF scores between generated and ground truth lyrics
    chrf, chrf_plus = get_chrf_scores(gt_lyrics, gen_lyrics)
    print('ChrF/ChrF++', chrf, chrf_plus)

    # Compute Corpus BLEU scores between generated and ground truth lyrics
    bleu_corpus = get_corpus_bleu_scores(gt_lyrics, gen_lyrics)
    print('Corpus BLEU 2/3/4/5: ', bleu_corpus)

    # Compute BERT scores between generated and ground truth lyrics
    bert_scores = get_bert_scores(gt_lyrics, gen_lyrics)
    print('Bert Scores: ', bert_scores)

    # Compute InfoLM scores (KL Divergence and L2 Distance) between generated and ground truth lyrics
    kl_divergence, l2_distance = get_infolm_scores(gt_lyrics, gen_lyrics)
    print('KL Divergence, L2 Distance: ', kl_divergence, l2_distance)

    print('Lyrics generated in ', lyrics_dir)
