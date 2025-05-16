"""
@Project: controllable-syllable-level-lyrics-generation-from-melody-with-prior-attention
@File: models.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
"""
# This module implements the Transformer-based melody-to-lyrics
# architecture, including encoders, decoders, discriminators, and
# beam-search translators with prior attention support.
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from transformer.Layers import EncoderLayer, DecoderLayer, DecoderLayerPrior
from utils import *


 # Mask out padding positions to prevent attention to padded tokens.
def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


 # Mask out future positions for autoregressive decoding.
def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for token embeddings."""

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class PositionalEncodingReturn(nn.Module):
    """Return-only positional encoding table for melody attributes."""

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncodingReturn, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        # return self.pos_table[:, :x.size(1)].clone().detach()
        return self.pos_table.clone().detach()

class Encoder(nn.Module):
    """Transformer encoder for token sequences."""

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # Embed tokens and add positional encodings.
        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,



class Decoder(nn.Module):
    """Transformer decoder for token sequences."""

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # Embed target tokens and prepare for self-attention.
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class MelodyEncoder(nn.Module):
    """Encoder for melody inputs: pitch, duration, and rest."""

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):

        super().__init__()

        # self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.src_pitch_emb = nn.Embedding(n_src_vocab[0], int(d_word_vec/4), padding_idx=pad_idx[0])
        self.src_duration_emb = nn.Embedding(n_src_vocab[1], int(d_word_vec/4), padding_idx=pad_idx[1])
        self.src_rest_emb = nn.Embedding(n_src_vocab[2], int(d_word_vec/4), padding_idx=pad_idx[2])

        self.position_enc = PositionalEncodingReturn(int(d_word_vec/4), n_position=n_position)

        self.melody_emb = nn.Linear(d_word_vec, d_word_vec)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):
        src_seq_pitch, src_seq_duration, src_seq_rest = src_seq

        enc_slf_attn_list = []

        # Embed pitch, duration, rest, concatenate, and project.
        emb_pitch = self.src_pitch_emb(src_seq_pitch)
        emb_duration = self.src_duration_emb(src_seq_duration)
        emb_rest = self.src_rest_emb(src_seq_rest)

        emb_position = self.position_enc(src_seq_pitch).expand_as(emb_pitch)
        emb_melody = torch.cat([emb_pitch, emb_duration, emb_rest, emb_position], dim=2)
        emb_melody = self.melody_emb(emb_melody)

        if self.scale_emb:
            emb_melody *= self.d_model ** 0.5

        enc_output = self.dropout(emb_melody)

        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class LyricsDecoder(nn.Module):
    """Decoder attending to melody representations for lyrics."""

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1):

        super().__init__()

        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list = []

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(enc_output, slf_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list
        return dec_output,


class LyricsDecoderTC(nn.Module):
    """Decoder with target-conditioned encoder-decoder attention."""

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output


class LyricsDecoderPrior(nn.Module):
    """Decoder variant incorporating prior attention constraints."""

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayerPrior(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask,
                slf_prior_attn=None, dec_enc_prior_attn=None, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask,
                slf_prior_attn=slf_prior_attn, dec_enc_prior_attn=dec_enc_prior_attn)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output


class TransformerWithNoise(nn.Module):
    """Seq2Seq model with optional noise injection into melody encoding."""

    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True,
            scale_emb_or_prj='none',
            enc_dec_attn=False):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb')
        self.scale_prj = (scale_emb_or_prj == 'prj')
        self.d_model = d_model
        self.enc_dec_attn = enc_dec_attn

        self.encoder = MelodyEncoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.noise_emb = nn.Linear(d_model, d_model, bias=True)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        if enc_dec_attn:
            self.decoder = LyricsDecoderTC(n_trg_vocab=n_trg_vocab, d_word_vec=d_word_vec,
                                           n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                                           d_model=d_model, d_inner=d_inner, pad_idx=trg_pad_idx,
                                           n_position=n_position, dropout=dropout, scale_emb=scale_emb)
            if trg_emb_prj_weight_sharing:
                # Share the weight between target word embedding & last dense layer
                self.trg_word_prj.weight = self.decoder.trg_word_emb.weight
        else:
            self.decoder = LyricsDecoder(
                n_position=n_position,
                d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
                n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                pad_idx=trg_pad_idx, dropout=dropout)
        # self.decoder = Decoder(n_trg_vocab=n_trg_vocab, d_word_vec=d_word_vec,
        #                        n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
        #                        d_model=d_model, d_inner=d_inner, pad_idx=trg_pad_idx,
        #                        n_position=n_position, dropout=dropout, scale_emb=scale_emb)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'


    def forward(self, src_seq, trg_seq, noise):

        src_seq_pitch, src_seq_duration, src_seq_rest = src_seq

        src_mask = get_pad_mask(src_seq_pitch, self.src_pad_idx[0])
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        # Encode melody attributes into latent representation.
        enc_output, *_ = self.encoder(src_seq, src_mask)

        if noise is not None:
            emb_noise = self.noise_emb(noise)
            z = enc_output + emb_noise
        else:
            z = enc_output

        if self.enc_dec_attn:
            dec_output = self.decoder(trg_seq, trg_mask, z, src_mask)
        else:
            dec_output = self.decoder(z, src_mask)

        seq_logit = self.trg_word_prj(dec_output)
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        return seq_logit



class TransformerWithNoiseMultiLevel(nn.Module):
    """Multi-level Seq2Seq model outputting syllable and word predictions."""

    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True,
            scale_emb_or_prj='none',
            enc_dec_attn=False):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb')
        self.scale_prj = (scale_emb_or_prj == 'prj')
        self.d_model = d_model
        self.enc_dec_attn = enc_dec_attn

        self.encoder = MelodyEncoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.noise_emb = nn.Linear(d_model, d_model, bias=True)

        self.trg_lyc_prj_syll = nn.Linear(d_model, n_trg_vocab[0], bias=False)
        self.trg_lyc_prj_word = nn.Linear(d_model, n_trg_vocab[1], bias=False)

        if enc_dec_attn:
            self.decoder = LyricsDecoderTC(n_trg_vocab=n_trg_vocab[0], d_word_vec=d_word_vec,
                                           n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                                           d_model=d_model, d_inner=d_inner, pad_idx=trg_pad_idx[0],
                                           n_position=n_position, dropout=dropout, scale_emb=scale_emb)
            if trg_emb_prj_weight_sharing:
                # Share the weight between target word embedding & last dense layer
                self.trg_lyc_prj_syll.weight = self.decoder.trg_word_emb.weight
        else:
            self.decoder = LyricsDecoder(
                n_position=n_position,
                d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
                n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                pad_idx=trg_pad_idx, dropout=dropout)
        # self.decoder = Decoder(n_trg_vocab=n_trg_vocab, d_word_vec=d_word_vec,
        #                        n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
        #                        d_model=d_model, d_inner=d_inner, pad_idx=trg_pad_idx,
        #                        n_position=n_position, dropout=dropout, scale_emb=scale_emb)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

    def forward(self, src_seq, trg_seq, noise, return_z=False):

        src_seq_pitch, src_seq_duration, src_seq_rest = src_seq

        src_mask = get_pad_mask(src_seq_pitch, self.src_pad_idx[0])
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx[0]) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)

        if noise is not None:
            emb_noise = self.noise_emb(noise)
            z = enc_output + emb_noise
        else:
            z = enc_output

        if self.enc_dec_attn:
            dec_output = self.decoder(trg_seq, trg_mask, z, src_mask)
        else:
            dec_output = self.decoder(z, src_mask)

        seq_logit_syll = self.trg_lyc_prj_syll(dec_output)
        seq_logit_word = self.trg_lyc_prj_word(dec_output)

        if self.scale_prj:
            seq_logit_syll *= self.d_model ** -0.5
            seq_logit_word *= self.d_model ** -0.5

        if return_z:
            return seq_logit_syll, seq_logit_word, z
        else:
            return seq_logit_syll, seq_logit_word

        # pred = self.softmax(seq_logit)
        # return pred


class TransformerWithPriorAttentionMultiLevel(nn.Module):
    """Multi-level Seq2Seq model with prior attention mechanism."""

    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True,
            scale_emb_or_prj='none',
            prior_attn=False):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb')
        self.scale_prj = (scale_emb_or_prj == 'prj')
        self.d_model = d_model
        self.prior_attn = prior_attn

        self.encoder = MelodyEncoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.noise_emb = nn.Linear(d_model, d_model, bias=True)

        self.trg_lyc_prj_syll = nn.Linear(d_model, n_trg_vocab[0], bias=False)
        self.trg_lyc_prj_word = nn.Linear(d_model, n_trg_vocab[1], bias=False)

        if prior_attn:
            self.decoder = LyricsDecoderPrior(n_trg_vocab=n_trg_vocab[0], d_word_vec=d_word_vec,
                                              n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                                              d_model=d_model, d_inner=d_inner, pad_idx=trg_pad_idx[0],
                                              n_position=n_position, dropout=dropout, scale_emb=scale_emb)
            if trg_emb_prj_weight_sharing:
                # Share the weight between target word embedding & last dense layer
                self.trg_lyc_prj_syll.weight = self.decoder.trg_word_emb.weight
        else:
            self.decoder = LyricsDecoderTC(n_trg_vocab=n_trg_vocab[0], d_word_vec=d_word_vec,
                                           n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                                           d_model=d_model, d_inner=d_inner, pad_idx=trg_pad_idx[0],
                                           n_position=n_position, dropout=dropout, scale_emb=scale_emb)
            if trg_emb_prj_weight_sharing:
                # Share the weight between target word embedding & last dense layer
                self.trg_lyc_prj_syll.weight = self.decoder.trg_word_emb.weight

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

    def forward(self, src_seq, trg_seq, prior_attn, return_z=False):

        src_seq_pitch, src_seq_duration, src_seq_rest = src_seq

        src_mask = get_pad_mask(src_seq_pitch, self.src_pad_idx[0])
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx[0]) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)

        # if noise is not None:
        #     emb_noise = self.noise_emb(noise)
        #     z = enc_output + emb_noise
        # else:
        #     z = enc_output

        z = enc_output

        if self.prior_attn:
            dec_output = self.decoder(trg_seq, trg_mask, z, src_mask,
                                      slf_prior_attn=prior_attn, dec_enc_prior_attn=None, return_attns=False)
        else:
            dec_output = self.decoder(trg_seq, trg_mask, z, src_mask)

        seq_logit_syll = self.trg_lyc_prj_syll(dec_output)
        seq_logit_word = self.trg_lyc_prj_word(dec_output)

        if self.scale_prj:
            seq_logit_syll *= self.d_model ** -0.5
            seq_logit_word *= self.d_model ** -0.5

        if return_z:
            return seq_logit_syll, seq_logit_word, z
        else:
            return seq_logit_syll, seq_logit_word


class CNNDiscriminator(nn.Module):
    """CNN discriminator for adversarial training on lyric sequences."""
    def __init__(self, embed_dim, vocab_size, filter_sizes, num_filters, padding_idx, dropout=0.2):
        super(CNNDiscriminator, self).__init__()
        self.embedding_dim = embed_dim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.feature_dim = sum(num_filters)
        # self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, embed_dim)) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, 2)
        self.dropout = nn.Dropout(dropout)

        self.init_params()

    def forward(self, inp):
        """
        Get final predictions of discriminator
        :param inp: batch_size * seq_len
        :return: pred: batch_size * 2
        """
        feature = self.get_feature(inp)
        pred = self.feature2out(self.dropout(feature))

        return pred

    def get_feature(self, inp):
        """
        Get feature vector of given sentences
        :param inp: batch_size * max_seq_len
        :return: batch_size * feature_dim
        """
        emb = self.embeddings(inp).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # tensor: batch_size * feature_dim
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway

        return pred

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                truncated_normal_(param, std=stddev)


class RelGAN_D(CNNDiscriminator):
    """Relational GAN discriminator for sequence evaluation."""
    # def __init__(self, embed_dim, max_seq_len, num_rep, vocab_size, padding_idx, dropout=0.25):
    def __init__(self, embed_dim, max_seq_len, num_rep, vocab_size,
                 dis_filter_sizes, dis_num_filters, padding_idx, dropout=0.25):
        super(RelGAN_D, self).__init__(embed_dim, vocab_size, dis_filter_sizes, dis_num_filters, padding_idx, dropout)

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.feature_dim = sum(dis_num_filters)
        self.emb_dim_single = int(embed_dim / num_rep)

        self.embeddings = nn.Linear(vocab_size, embed_dim, bias=False)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, self.emb_dim_single), stride=(1, self.emb_dim_single)) for (n, f) in
            zip(dis_num_filters, dis_filter_sizes)
        ])

        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, 100)
        self.out2logits = nn.Linear(100, 1)
        self.dropout = nn.Dropout(dropout)

        self.init_params()

    def forward(self, inp):
        """
        Get logits of discriminator
        :param inp: batch_size * seq_len * vocab_size
        :return logits: [batch_size * num_rep] (1-D tensor)
        """
        emb = self.embeddings(inp).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim

        cons = [F.relu(conv(emb)) for conv in self.convs]  # [batch_size * num_filter * (seq_len-k_h+1) * num_rep]
        pools = [F.max_pool2d(con, (con.size(2), 1)).squeeze(2) for con in cons]  # [batch_size * num_filter * num_rep]
        pred = torch.cat(pools, 1)
        pred = pred.permute(0, 2, 1).contiguous().view(-1, self.feature_dim)  # (batch_size * num_rep) * feature_dim
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway

        pred = self.feature2out(self.dropout(pred))
        logits = self.out2logits(pred).squeeze(1)  # [batch_size * num_rep]

        return logits


class ConditionalRelGAN_D(CNNDiscriminator):
    """Conditional relational GAN discriminator with melody context."""
    # def __init__(self, embed_dim, max_seq_len, num_rep, vocab_size, padding_idx, dropout=0.25):
    def __init__(self, embed_dim, max_seq_len, num_rep, vocab_size,
                 dis_filter_sizes, dis_num_filters, padding_idx,
                 condition_sizes, condition_paddings, dropout=0.25):
        super(ConditionalRelGAN_D, self).__init__(embed_dim, vocab_size, dis_filter_sizes, dis_num_filters, padding_idx, dropout)

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.feature_dim = sum(dis_num_filters)
        self.emb_dim_single = int(embed_dim / num_rep)

        self.embeddings = nn.Linear(vocab_size, embed_dim, bias=False)
        self.con_pitch_emb = nn.Embedding(condition_sizes[0], embed_dim, padding_idx=condition_paddings[0])
        self.con_duration_emb = nn.Embedding(condition_sizes[1], embed_dim, padding_idx=condition_paddings[1])
        self.con_rest_emb = nn.Embedding(condition_sizes[2], embed_dim, padding_idx=condition_paddings[2])

        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, self.emb_dim_single), stride=(1, self.emb_dim_single)) for (n, f) in
            zip(dis_num_filters, dis_filter_sizes)
        ])

        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, 100)
        self.out2logits = nn.Linear(100, 1)
        self.dropout = nn.Dropout(dropout)

        self.init_params()

    def forward(self, inp, condition):
        """
        Get logits of discriminator
        :param inp: batch_size * seq_len * vocab_size
        :return logits: [batch_size * num_rep] (1-D tensor)
        """
        emb_lyrics = self.embeddings(inp).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim

        emb_pitch = self.con_pitch_emb(condition[0]).unsqueeze(1)
        emb_duration = self.con_duration_emb(condition[1]).unsqueeze(1)
        emb_rest = self.con_rest_emb(condition[2]).unsqueeze(1)

        emb = emb_lyrics + emb_pitch + emb_duration + emb_rest

        cons = [F.relu(conv(emb)) for conv in self.convs]  # [batch_size * num_filter * (seq_len-k_h+1) * num_rep]
        pools = [F.max_pool2d(con, (con.size(2), 1)).squeeze(2) for con in cons]  # [batch_size * num_filter * num_rep]
        pred = torch.cat(pools, 1)
        pred = pred.permute(0, 2, 1).contiguous().view(-1, self.feature_dim)  # (batch_size * num_rep) * feature_dim
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway

        pred = self.feature2out(self.dropout(pred))
        logits = self.out2logits(pred).squeeze(1)  # [batch_size * num_rep]

        return logits


class Translator(nn.Module):
    """Beam-search translator for autoregressive lyric decoding."""

    def __init__(
            self, model, beam_size, max_seq_len,
            src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):

        super(Translator, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        self.register_buffer(
            'blank_seqs',
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.register_buffer(
            'len_map',
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))

    def _model_decode(self, trg_seq, enc_output, src_mask):
        trg_mask = get_subsequent_mask(trg_seq)
        dec_output = self.model.decoder(trg_seq, trg_mask, enc_output, src_mask)
        if hasattr(self.model, 'trg_lyc_prj_syll'):
            return F.softmax(self.model.trg_lyc_prj_syll(dec_output), dim=-1)
        elif hasattr(self.model, 'trg_word_prj'):
            return F.softmax(self.model.trg_word_prj(dec_output), dim=-1)
        else:
            raise Exception("Wrong decoder projection layer.")

    def _get_init_state(self, src_seq, src_mask, noise=None):
        beam_size = self.beam_size

        enc_output, *_ = self.model.encoder(src_seq, src_mask)

        if noise is not None:
            enc_output = enc_output + noise

        dec_output = self._model_decode(self.init_seq, enc_output, src_mask)

        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)

        scores = torch.log(best_k_probs).view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        enc_output = enc_output.repeat(beam_size, 1, 1)
        return enc_output, gen_seq, scores

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1

        beam_size = self.beam_size

        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)

        # Include the previous scores.
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)

        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)

        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        # Copy the corresponding previous tokens.
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores

    def translate_sentence(self, src_seq, noise=None):
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        # assert src_seq.size(0) == 3

        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha

        src_seq_pitch, src_seq_duration, src_seq_rest = src_seq

        with torch.no_grad():
            # Perform beam search decoding with length normalization.
            # src_mask = get_pad_mask(src_seq, src_pad_idx)
            src_mask = get_pad_mask(src_seq_pitch, src_pad_idx[0])

            enc_output, gen_seq, scores = self._get_init_state(src_seq, src_mask, noise=noise)

            ans_idx = 0  # default
            for step in range(2, max_seq_len):  # decode up to max length
                dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_mask)
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)

                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == trg_eos_idx
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)

                # We used same token for padding and eos...

                # -- check if all beams contain eos
                # if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                #     # TODO: Try different terminate conditions.
                #     _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                #     ans_idx = ans_idx.item()
                #     break

            _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
            ans_idx = ans_idx.item()
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()


class TranslatorPrior(nn.Module):
    """Beam-search translator using prior attention during decoding."""

    def __init__(
            self, model, beam_size, max_seq_len,
            src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):

        super(TranslatorPrior, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        self.register_buffer(
            'blank_seqs',
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.register_buffer(
            'len_map',
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))

    def _model_decode(self, trg_seq, enc_output, src_mask, prior_attn=None):
        trg_mask = get_subsequent_mask(trg_seq)

        dec_output = self.model.decoder(trg_seq, trg_mask, enc_output, src_mask,
                                        slf_prior_attn=prior_attn, dec_enc_prior_attn=None, return_attns=False)
        if hasattr(self.model, 'trg_lyc_prj_syll'):
            return F.softmax(self.model.trg_lyc_prj_syll(dec_output), dim=-1)
        elif hasattr(self.model, 'trg_word_prj'):
            return F.softmax(self.model.trg_word_prj(dec_output), dim=-1)
        else:
            raise Exception("Wrong decoder projection layer.")

    def _get_init_state(self, src_seq, src_mask, noise=None, prior_attn=None):
        beam_size = self.beam_size

        enc_output, *_ = self.model.encoder(src_seq, src_mask)

        if noise is not None:
            enc_output = enc_output + noise

        dec_output = self._model_decode(self.init_seq, enc_output, src_mask)

        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)

        scores = torch.log(best_k_probs).view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        enc_output = enc_output.repeat(beam_size, 1, 1)
        return enc_output, gen_seq, scores

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1

        beam_size = self.beam_size

        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)

        # Include the previous scores.
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)

        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)

        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        # Copy the corresponding previous tokens.
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores

    def translate_sentence(self, src_seq, noise=None, prior_attn=None):
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        # assert src_seq.size(0) == 3

        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha

        src_seq_pitch, src_seq_duration, src_seq_rest = src_seq

        with torch.no_grad():
            # src_mask = get_pad_mask(src_seq, src_pad_idx)
            src_mask = get_pad_mask(src_seq_pitch, src_pad_idx[0])

            enc_output, gen_seq, scores = self._get_init_state(src_seq, src_mask, noise=noise, prior_attn=prior_attn)

            ans_idx = 0  # default
            for step in range(2, max_seq_len):  # decode up to max length

                dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_mask,
                                                prior_attn=prior_attn[:, :step, :step])
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)

                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == trg_eos_idx
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)

                # We used same token for padding and eos...

                # -- check if all beams contain eos
                # if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                #     # TODO: Try different terminate conditions.
                #     _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                #     ans_idx = ans_idx.item()
                #     break

            _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
            ans_idx = ans_idx.item()
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()
