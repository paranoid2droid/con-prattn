"""
@Project: controllable-syllable-level-lyrics-generation-from-melody-with-prior-attention
@File: preprocess.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
"""

# This script preprocesses paired melody and lyrics data for the controllable
# syllable-level lyrics generation task. It builds fixed-length sequences,
# computes flags, tokenizes text and musical attributes, and saves processed
# numpy datasets and tokenizers for training.

import os

import keras_preprocessing.text
import numpy as np
import pickle
from tqdm import tqdm
import tensorflow as tf
# https://stackoverflow.com/questions/71000250/import-tensorflow-keras-could-not-be-resolved-after-upgrading-to-tensorflow-2/71838765#71838765
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.api._v2.keras.preprocessing.text import Tokenizer
from keras.api._v2.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Define the path to raw parsed song files and list all files.
songs_path = './data/lmd-full_MIDI_dataset/Sentence_and_Word_Parsing'
# songs_path = './data/lmd-full_MIDI_dataset/Syllable_Parsing'
songs = os.listdir(songs_path)

# Set maximum sequence length (excluding BOS/EOS) for concatenated phrases.
max_len = 100

# Load all song numpy features into memory.
features = []
for song in tqdm(songs):
    features.append(np.load(f"{songs_path}/{song}", allow_pickle=True))

# Extract melody attribute arrays: pitch, duration, rest from loaded features.
melody = np.array(features)[:, :, 1]
lyrics_word = np.array(features)[:, :, 2]
lyrics_syllable = np.array(features)[:, :, 3]

lyrics_word_list = []
lyrics_syllable_list = []
lyrics_length_list = []


notes_list, durations_list, rests_list = [], [], []
position_flag_list = []

# Compute phrase lengths for each song for analysis (optional).
song_phrase_length = []
for song_syllable in tqdm(lyrics_syllable):
    phrase_length = []
    for phrase_syllable in song_syllable[0]:
        phrase_syllable = sum(phrase_syllable, [])

        phrase_length.append(len(phrase_syllable))

    song_phrase_length.append(phrase_length)

# Iterate through songs to flatten phrases and build word, syllable, melody, and flag lists.
for song_word, song_syllable, song_melody in tqdm(zip(lyrics_word, lyrics_syllable, melody)):
    word_list = []
    syllable_list = []
    melody_list = []
    flag_list = []

    for idx, (phrase_word, phrase_syllable, phrase_melody) in \
            enumerate(zip(song_word[0], song_syllable[0], song_melody[0])):

        assert len(phrase_word) == len(phrase_syllable) == len(phrase_melody)
        if not phrase_word:
            continue

        phrase_flag = []
        for phrase_word_item in phrase_word:
            assert len(phrase_word_item) > 0
            if len(phrase_word_item) == 1:
                # single syllable
                phrase_flag.append(0)
            else:
                for i, item in enumerate(phrase_word_item):
                    # begin of word
                    if i == 0:
                        phrase_flag.append(1)
                    # end of word
                    elif i == len(phrase_word_item) - 1:
                        phrase_flag.append(2)
                    # middle of word
                    else:
                        phrase_flag.append(3)

        phrase_word_flat = sum(phrase_word, [])
        phrase_syllable_flat = sum(phrase_syllable, [])
        phrase_melody_flat = [n for word_level in phrase_melody for n in word_level]

        if len(word_list) + len(phrase_word_flat) <= max_len:
            word_list.extend(phrase_word_flat)
            syllable_list.extend(phrase_syllable_flat)
            melody_list.extend(phrase_melody_flat)
            flag_list.extend(phrase_flag)
            if idx == len(song_word[0]) - 1:
                if len(word_list) < 2:
                    continue
                # save
                assert len(word_list) == len(syllable_list) == len(melody_list) == len(flag_list)
                words = "<bos> " + " ".join(word_list) + " <eos>"
                syllables = "<bos> " + " ".join(syllable_list) + " <eos>"

                note = np.array(melody_list)[:, 0]
                duration = np.array(melody_list)[:, 1]
                rest = np.array(melody_list)[:, 2]
                note_str = "<bos> " + " ".join([str(n) for n in note]) + " <eos>"
                duration_str = "<bos> " + " ".join([str(d) for d in duration]) + " <eos>"
                rest_str = "<bos> " + " ".join([str(r) for r in rest]) + " <eos>"

                flags = np.pad(np.array([-1] + flag_list + [-1]), (0, max_len - len(flag_list)), constant_values=-1)

                lyrics_word_list.append(words)
                lyrics_syllable_list.append(syllables)
                notes_list.append(note_str)
                durations_list.append(duration_str)
                rests_list.append(rest_str)
                lyrics_length_list.append(len(syllable_list))

                position_flag_list.append(flags)

                # initialize
                word_list = []
                syllable_list = []
                melody_list = []
                flag_list = []
        else:  # len(word_list) + len(phrase_word_flat) > max_len
            # save
            if not word_list:
                word_list.extend(phrase_word_flat)
                syllable_list.extend(phrase_syllable_flat)
                melody_list.extend(phrase_melody_flat)
                flag_list.extend(phrase_flag)
            if len(word_list) > max_len:
                # word_list = word_list[:max_len]
                # syllable_list = syllable_list[:max_len]
                # melody_list = melody_list[:max_len]
                # flag_list = flag_list[:max_len]
                continue
            if len(word_list) < 2:
                continue

            assert len(word_list) == len(syllable_list) == len(melody_list) == len(flag_list)

            words = "<bos> " + " ".join(word_list) + " <eos>"
            syllables = "<bos> " + " ".join(syllable_list) + " <eos>"

            note = np.array(melody_list)[:, 0]
            duration = np.array(melody_list)[:, 1]
            rest = np.array(melody_list)[:, 2]
            note_str = "<bos> " + " ".join([str(n) for n in note]) + " <eos>"
            duration_str = "<bos> " + " ".join([str(d) for d in duration]) + " <eos>"
            rest_str = "<bos> " + " ".join([str(r) for r in rest]) + " <eos>"

            flags = np.pad(np.array([-1] + flag_list + [-1]), (0, max_len - len(flag_list)), constant_values=-1)

            lyrics_word_list.append(words)
            lyrics_syllable_list.append(syllables)
            notes_list.append(note_str)
            durations_list.append(duration_str)
            rests_list.append(rest_str)
            lyrics_length_list.append(len(syllable_list))

            position_flag_list.append(flags)

            # initialize
            # word_list = []
            # syllable_list = []
            # melody_list = []
            # flag_list = []
            # start from the current phrase
            word_list = phrase_word_flat
            syllable_list = phrase_syllable_flat
            melody_list = phrase_melody_flat
            flag_list = phrase_flag

# Visualize the distribution of sequence lengths (optional, requires display).
fig = plt.subplots(dpi=300)
counts, _, patches = plt.hist(lyrics_length_list, bins=10, range=(0, 100))
for count, patch in zip(counts, patches):
    plt.annotate(str(int(count)), xy=(patch.get_x()+0.2, patch.get_height()+100))
# plt.hist(length_list)
plt.xlabel('Lyrics Sequence Length', fontsize=12)
plt.ylabel('Sequence Amount', fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
plt.tight_layout()
plt.show()

# Initialize and fit tokenizer for syllable-level tokens.
tokenizer_syllable = Tokenizer(oov_token="<unk>", filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
tokenizer_syllable.fit_on_texts(lyrics_syllable_list)

# Initialize and fit tokenizer for word-level tokens.
tokenizer_word = Tokenizer(oov_token="<unk>", filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
tokenizer_word.fit_on_texts(lyrics_word_list)

print('Syllable count: ', tokenizer_syllable.word_counts)
print('Word count: ', tokenizer_word.word_counts)

pad_id_syllable = tokenizer_syllable.word_index["<eos>"]
start_id_syllable = tokenizer_syllable.word_index["<bos>"]

sequences_syllable_tokens = tokenizer_syllable.texts_to_sequences(lyrics_syllable_list)
sequences_syllable_tokens = pad_sequences(sequences_syllable_tokens,
                                          maxlen=max_len+2, truncating='post', padding='post', value=pad_id_syllable)

pad_id_word = tokenizer_word.word_index["<eos>"]
start_id_word = tokenizer_word.word_index["<bos>"]

sequences_word_tokens = tokenizer_word.texts_to_sequences(lyrics_word_list)
sequences_word_tokens = pad_sequences(sequences_word_tokens,
                                      maxlen=max_len + 2, truncating='post', padding='post', value=pad_id_word)

# Initialize and fit tokenizer for note sequences.
tokenizer_note = Tokenizer(filters='!"#$%&()*+,-/:;=?@[\\]^_`{|}~\t\n')
tokenizer_note.fit_on_texts(notes_list)

pad_id_note = tokenizer_note.word_index["<eos>"]
start_id_note = tokenizer_note.word_index["<bos>"]

sequences_note = tokenizer_note.texts_to_sequences(notes_list)
sequences_note = pad_sequences(sequences_note,
                               truncating='post', padding='post', value=pad_id_note)

# Initialize and fit tokenizer for duration sequences.
tokenizer_duration = Tokenizer(filters='!"#$%&()*+,-/:;=?@[\\]^_`{|}~\t\n')
tokenizer_duration.fit_on_texts(durations_list)

pad_id_duration = tokenizer_duration.word_index["<eos>"]
start_id_duration = tokenizer_duration.word_index["<bos>"]

sequences_duration = tokenizer_duration.texts_to_sequences(durations_list)
sequences_duration = pad_sequences(sequences_duration,
                                   truncating='post', padding='post', value=pad_id_duration)

# Initialize and fit tokenizer for rest sequences.
tokenizer_rest = Tokenizer(filters='!"#$%&()*+,-/:;=?@[\\]^_`{|}~\t\n')
tokenizer_rest.fit_on_texts(rests_list)

pad_id_rest = tokenizer_rest.word_index["<eos>"]
start_id_rest = tokenizer_rest.word_index["<bos>"]

sequences_rest = tokenizer_rest.texts_to_sequences(rests_list)
sequences_rest = pad_sequences(sequences_rest,
                               truncating='post', padding='post', value=pad_id_rest)

sequences_syllable = np.expand_dims(sequences_syllable_tokens, axis=2)
sequences_word = np.expand_dims(sequences_word_tokens, axis=2)

sequences_note = np.expand_dims(sequences_note, axis=2)
sequences_duration = np.expand_dims(sequences_duration, axis=2)
sequences_rest = np.expand_dims(sequences_rest, axis=2)

sequences_flag = np.expand_dims(np.array(position_flag_list), axis=2)

sequences = np.concatenate([sequences_syllable, sequences_word, sequences_note, sequences_duration, sequences_rest,
                            sequences_flag], axis=2)
# sequences = np.concatenate([sequences_syllable, sequences_note, sequences_duration, sequences_rest], axis=2)

sequences, duplicated_idx = np.unique(sequences, axis=0, return_index=True)
sequences_unique = np.unique(sequences, axis=0)

assert len(sequences_unique) == len(sequences)

data_folder = './data/syllable_and_word_100_flag_clean'

tokenizer_folder = os.path.join(data_folder, 'tokenizers')

if not os.path.exists(data_folder):
    os.mkdir(data_folder)

if not os.path.exists(tokenizer_folder):
    os.mkdir(tokenizer_folder)

# Utility function to save a tokenizer object to disk.
def save_tokenizer(file, tokenizer):
    with open(file, 'wb') as handle:
        pickle.dump({'tokenizer': tokenizer}, handle)

save_tokenizer(os.path.join(tokenizer_folder, 'tokenizer_syllable.pkl'), tokenizer_syllable)
save_tokenizer(os.path.join(tokenizer_folder, 'tokenizer_word.pkl'), tokenizer_word)

save_tokenizer(os.path.join(tokenizer_folder, 'tokenizer_note.pkl'), tokenizer_note)
save_tokenizer(os.path.join(tokenizer_folder, 'tokenizer_duration.pkl'), tokenizer_duration)
save_tokenizer(os.path.join(tokenizer_folder, 'tokenizer_rest.pkl'), tokenizer_rest)

# notes_size = len(tokenizer_note.word_index) + 1
# durations_size = len(tokenizer_duration.word_index) + 1
# rests_size = len(tokenizer_rest.word_index) + 1

# Split unique sequences into train, validation, and test sets (8:1:1 ratio).
num_sequences = len(sequences)
val_test_len = int(num_sequences * 0.1) * 2
# train_inds = np.random.choice(np.arange(num_songs), size=num_samples, replace=False)
# test_inds = np.delete(np.arange(num_songs), train_inds)

val_test_inds = np.random.choice(np.arange(num_sequences), size=val_test_len, replace=False)
train_inds = np.delete(np.arange(num_sequences), val_test_inds)
valid_inds = np.random.choice(val_test_inds, size=int(val_test_len/2), replace=False)
test_inds = np.delete(val_test_inds, list(np.where(val_test_inds == idx) for idx in valid_inds))

sequences_train = sequences[train_inds]
sequences_valid = sequences[valid_inds]
sequences_test = sequences[test_inds]

print('length of train, valid, and test data: ', len(sequences_train), len(sequences_valid), len(sequences_test))

# sequences_train = sequences[:num_sequences - val_test_len]
# sequences_val_test = sequences[num_sequences - val_test_len:]
# valid_inds = np.random.choice(np.arange(val_test_len), size=int(val_test_len/2), replace=False)
# test_inds = np.delete(np.arange(val_test_len), valid_inds)
# sequences_valid = sequences_val_test[valid_inds]
# sequences_test = sequences_val_test[test_inds]

# Save processed sequences and dataset splits to disk.
np.save(os.path.join(data_folder, 'sequences.npy'), sequences)
np.save(os.path.join(data_folder, 'sequences_train.npy'), sequences_train)
np.save(os.path.join(data_folder, 'sequences_valid.npy'), sequences_valid)
np.save(os.path.join(data_folder, 'sequences_test.npy'), sequences_test)
