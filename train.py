"""
@Project: controllable-syllable-level-lyrics-generation-from-melody-with-prior-attention
@File: train.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
"""

# This script sets up data loading, model initialization, and training for
# the controllable syllable-level lyrics generation task using PyTorch Lightning.

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from transformer_multi_regularization_flag import TransformerGeneratorRegularizationFlag
from dataset_features_flags import LyricsMelodyDataset, LyricsMelodyDataModule
from hparams import *

from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary


# Set random seed for reproducibility
pl.seed_everything(42)

# Ensure deterministic behavior on GPU for reproducibility
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configure device: use GPU if available, else CPU
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

# Utility to print total and trainable parameter counts of the model
def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")


# Main entry point: prepare data, model, and trainer, then run training
if __name__ == '__main__':

    # Define base folder for data and tokenizers
    data_folder = ''
    tokenizer_folder = os.path.join(data_folder, 'tokenizers')

    # Load full dataset for vocabulary and token index retrieval
    lyrics_melody_dataset = LyricsMelodyDataset(data=os.path.join(data_folder, 'sequences.npy'),
                                                tokenizer_syllable_path=os.path.join(tokenizer_folder, 'tokenizer_syllable.pkl'),
                                                tokenizer_word_path=os.path.join(tokenizer_folder, 'tokenizer_word.pkl'),
                                                tokenizer_pitch_path=os.path.join(tokenizer_folder, 'tokenizer_note.pkl'),
                                                tokenizer_duration_path=os.path.join(tokenizer_folder, 'tokenizer_duration.pkl'),
                                                tokenizer_rest_path=os.path.join(tokenizer_folder, 'tokenizer_rest.pkl')
                                                )

    # Initialize data module for training, validation, and test splits
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

    # Create PyTorch DataLoaders from the data module
    train_loader = lyrics_melody_data_module.train_dataloader()
    val_loader = lyrics_melody_data_module.val_dataloader()
    test_loader = lyrics_melody_data_module.test_dataloader()

    # Retrieve vocabulary sizes and special token indices from dataset
    max_seq_len = MAX_LEN + 1

    vocab_size_syllable, vocab_size_word, vocab_size_pitch, vocab_size_duration, vocab_size_rest = lyrics_melody_dataset.get_vocab_size()
    padding_idx_syllable, padding_idx_word, padding_idx_pitch, padding_idx_duration, padding_idx_rest = lyrics_melody_dataset.get_padding_ids()
    start_id_syllable, start_id_word, start_id_pitch, start_id_duration, start_id_rest = lyrics_melody_dataset.get_start_ids()

    # Prepare directory for saving model checkpoints
    root_dir = './checkpoints'
    os.makedirs(root_dir, exist_ok=True)

    # Load a pretrained model checkpoint if it exists, else set up training
    # pretrained_gen_filename = os.path.join(CHECKPOINT_PATH, "test.ckpt")
    pretrained_gen_filename = 'None'
    if os.path.isfile(pretrained_gen_filename):
        print("Found pretrained model, loading...")
        model_gen = TransformerGeneratorRegularizationFlag.load_from_checkpoint(pretrained_gen_filename)
    else:
        # Set up Neptune logger, learning rate monitor, and model summary callback
        neptune_logger = NeptuneLogger(
            project="",
            api_key="",
            name='',
            log_model_checkpoints=False,
            mode="debug",  # "async", "sync", "offline", "read-only", or "debug"
        )

        lr_monitor = LearningRateMonitor(logging_interval='step')

        # Configure PyTorch Lightning Trainer with callbacks and training parameters
        trainer_gen = pl.Trainer(
            logger=neptune_logger,
            default_root_dir=root_dir,
            # callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
            gpus=1 if str(device).startswith("cuda") else 0,
            max_epochs=3000,
            gradient_clip_val=5,
            check_val_every_n_epoch=20,
            enable_model_summary=True,
            callbacks=[
                ModelCheckpoint(filename='pre-{epoch}-{step}', save_top_k=-1, every_n_epochs=100, save_last=True),
                lr_monitor,
                ModelSummary(max_depth=-1)],
        )
        trainer_gen.logger._default_hp_metric = None  # Optional logging argument that we don't need

        # Initialize model with hyperparameters and tokenizer paths
        model_gen = TransformerGeneratorRegularizationFlag(embedding_dim=EMBEDDING_DIM,
                                                           hidden_dim=HIDDEN_DIM,
                                                           inner_dim=INNER_DIM,
                                                           num_heads=NUM_HEADS,
                                                           num_layers=NUM_LAYERS,
                                                           d_k=D_K,
                                                           d_v=D_V,
                                                           syllable_vocab_size=vocab_size_syllable,
                                                           word_vocab_size=vocab_size_word,
                                                           pitch_size=vocab_size_pitch,
                                                           duration_size=vocab_size_duration,
                                                           rest_size=vocab_size_rest,
                                                           max_seq_len=max_seq_len,
                                                           bos_idx_syllable=start_id_syllable,
                                                           pad_idx_syllable=padding_idx_syllable,
                                                           pad_idx_word=padding_idx_word,
                                                           pad_idx_pitch=padding_idx_pitch,
                                                           pad_idx_duration=padding_idx_duration,
                                                           pad_idx_rest=padding_idx_rest,
                                                           dropout=DROPOUT,
                                                           lr=G_LR_PRE,
                                                           warmup=200,
                                                           max_iters=trainer_gen.max_epochs * len(train_loader),
                                                           tokenizer_syllable_path=os.path.join(tokenizer_folder,
                                                                                                'tokenizer_syllable.pkl'),
                                                           tokenizer_word_path=os.path.join(tokenizer_folder,
                                                                                            'tokenizer_word.pkl'),
                                                           label_smoothing=0,
                                                           n_gram_alpha=0.2,
                                                           prior_attn=False,
                                                           diagonal_fill=False,
                                                           lr_scheduler=False,
                                                           loss_crossfade=False,
                                                           multi_level=False)

        print_model_parameters(model_gen)
        # Start training the model
        trainer_gen.fit(model_gen, train_loader, val_loader)
