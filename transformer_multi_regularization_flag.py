"""
@Project: controllable-syllable-level-lyrics-generation-from-melody-with-prior-attention
@File: transformer_multi_regularization_flag.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
"""


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from models import *
import torch.optim as optim
from utils import *
from evaluation import get_bert_scores, get_rouge_scores, get_bleu_scores
import torch.nn.functional as F
from itertools import repeat
from n_gram_criterion import ExplicitNGramCriterion
from torchmetrics.classification import MulticlassAccuracy

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


class TransformerGeneratorRegularizationFlag(pl.LightningModule):
    def __init__(
            self,
            embedding_dim,
            hidden_dim,
            inner_dim,
            num_heads,
            num_layers,
            d_k,
            d_v,
            syllable_vocab_size,
            word_vocab_size,
            pitch_size,
            duration_size,
            rest_size,
            max_seq_len,
            bos_idx_syllable,
            pad_idx_syllable,
            pad_idx_word,
            pad_idx_pitch,
            pad_idx_duration,
            pad_idx_rest,
            dropout,
            lr,
            warmup=100,
            max_iters=10000,
            label_smoothing=0,
            n_gram_alpha=0,
            tokenizer_syllable_path=None,
            tokenizer_word_path=None,
            prior_attn=False,
            diagonal_fill=False,
            lr_scheduler=False,
            loss_crossfade=False,
            multi_level=False
    ):

        super().__init__()
        self.save_hyperparameters()
        self._create_model()

    def _create_model(self):
        self.generator = TransformerWithPriorAttentionMultiLevel(
            n_src_vocab=[self.hparams.pitch_size, self.hparams.duration_size, self.hparams.rest_size],
            n_trg_vocab=[self.hparams.syllable_vocab_size, self.hparams.word_vocab_size],
            src_pad_idx=[self.hparams.pad_idx_pitch, self.hparams.pad_idx_duration, self.hparams.pad_idx_rest],
            trg_pad_idx=[self.hparams.pad_idx_syllable, self.hparams.pad_idx_word],
            d_word_vec=self.hparams.embedding_dim,
            d_model=self.hparams.hidden_dim,
            d_inner=self.hparams.inner_dim,
            n_layers=self.hparams.num_layers,
            n_head=self.hparams.num_heads,
            d_k=self.hparams.d_k,
            d_v=self.hparams.d_v,
            dropout=self.hparams.dropout,
            n_position=self.hparams.max_seq_len,
            prior_attn=self.hparams.prior_attn,)

        if self.hparams.n_gram_alpha > 0:
            self.explicit_n_gram_loss = ExplicitNGramCriterion()

        self.alpha = 0.0

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        if self.hparams.lr_scheduler:
            self.lr_scheduler = CosineWarmupScheduler(
                optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters
            )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.hparams.lr_scheduler:
            self.lr_scheduler.step()  # Step per iteration

    def _cal_loss(self, pred, target, label_smoothing=0.0):
        pred_flat = pred.contiguous().view(-1, pred.size(-1))

        target_flat = target.contiguous().view(-1)

        loss = F.cross_entropy(pred_flat, target_flat, label_smoothing=label_smoothing)

        return loss

    def _cal_explicit_ngram_loss(self, pred, target, flags, padding_idx=None):
        loss = self.explicit_n_gram_loss(pred, target, flags, padding_idx=padding_idx)
        return loss

    def _cal_accuracy(self, pred, gold, padding_idx):
        gold_masked = torch.where(gold != padding_idx, gold, -1).to(device)
        eq_count = (gold_masked == pred).sum()

        bs = pred.shape[0]
        seq_len = self.hparams.max_seq_len
        acc = eq_count / bs / seq_len

        return acc

    def _get_pad_mask(self, seq, pad_idx):
        return (seq != pad_idx).unsqueeze(-1)

    def _generate_block_diagonal(self, flags_batch, diagonal_fill=False):
        batch_size = flags_batch.size(0)
        max_length = flags_batch.size(1)

        output_tensor_batch = torch.zeros((batch_size, max_length, max_length), dtype=torch.int)

        begin_flags = (flags_batch == 1).nonzero(as_tuple=False)
        end_flags = (flags_batch == 2).nonzero(as_tuple=False)

        for begin, end in zip(begin_flags, end_flags):
            # if not (begin[0] == end[0] and begin[1] < end[1]):
            #     torch.set_printoptions(profile="full")
            #     print(flags_batch)
            #     print("begin: {}, end: {}".format(begin, end))
            assert begin[0] == end[0] and begin[1] < end[1], "begin: {}, end: {}".format(begin, end)
            output_tensor_batch[begin[0], begin[1]:end[1] + 1, begin[1]:end[1] + 1] = 1
            torch.set_printoptions(profile="full")
        # print(output_tensor_batch[0])

        if diagonal_fill:
            # Create a mask tensor with 1s on the diagonal
            diagonal_mask = torch.eye(max_length, dtype=torch.int).unsqueeze(0).expand(batch_size, -1, -1)
            # Apply the mask to the original tensor
            output_tensor_batch = output_tensor_batch * (1 - diagonal_mask) + diagonal_mask
        # print(output_tensor_batch[0])

        return output_tensor_batch

    def forward(self, seq_pitch, seq_duration, seq_rest, target_syllable, target_word,
                position_flags=None, return_z=False):

        src_seq_pitch = seq_pitch[:, :-1]
        src_seq_duration = seq_duration[:, :-1]
        src_seq_rest = seq_rest[:, :-1]
        trg_seq_syllable = target_syllable[:, :-1]

        gold_syllable = target_syllable[:, 1:]
        gold_word = target_word[:, 1:]

        src_seq = src_seq_pitch.to(device), src_seq_duration.to(device), src_seq_rest.to(device)
        trg_seq_syllable = trg_seq_syllable.to(device)

        assert src_seq_pitch.shape == src_seq_duration.shape == src_seq_rest.shape == trg_seq_syllable.shape == \
               gold_syllable.shape == gold_word.shape

        if self.hparams.prior_attn:
            position_flags = position_flags[:, 1:]
            # position_flags = position_flags[:, :-1]
            prior_attention_masks = self._generate_block_diagonal(position_flags,
                                                                  diagonal_fill=self.hparams.diagonal_fill).to(device)
            # torch.set_printoptions(profile="full")
            # print(position_flags[0])
            # print(prior_attention_masks[0])
        else:
            prior_attention_masks = None

        pred_syllable, pred_word, z = self.generator.forward(src_seq, trg_seq_syllable,
                                                             prior_attn=prior_attention_masks, return_z=return_z)

        return (pred_syllable, pred_word), (gold_syllable, gold_word), z

    def training_step(self, batch, batch_idx):
        seq_pitch, seq_duration, seq_rest = batch[0]
        target_syllable, target_word = batch[1]
        position_flags = batch[2]
        feature_coordinates = batch[3]

        bs = target_syllable.shape[0]
        seq_len = self.hparams.max_seq_len

        pred, gold, z = self.forward(seq_pitch, seq_duration, seq_rest, target_syllable, target_word,
                                     position_flags=position_flags, return_z=True)
        (pred_syllable, pred_word), (gold_syllable, gold_word) = pred, gold

        gen_samples_syllable = F.softmax(pred_syllable, dim=-1)
        gen_samples_word = F.softmax(pred_word, dim=-1)

        real_samples_syllable = F.one_hot(gold_syllable, self.hparams.syllable_vocab_size).float()
        real_samples_word = F.one_hot(gold_word, self.hparams.word_vocab_size).float()

        if (self.current_epoch + 1) % 20 == 0:
            if batch_idx == 0:
                gt_lyrics_syllable = sequences_to_texts(real_samples_syllable, tokenizer_lyr_path=self.hparams.tokenizer_syllable_path)
                self.logger.experiment["lyrics/train_gt"].log(gt_lyrics_syllable[0])
                gt_lyrics_word = sequences_to_texts(real_samples_word, tokenizer_lyr_path=self.hparams.tokenizer_word_path)
                self.logger.experiment["lyrics/train_gt"].log(gt_lyrics_word[0])

                gen_lyrics_syllable = sequences_to_texts(gen_samples_syllable, tokenizer_lyr_path=self.hparams.tokenizer_syllable_path)
                self.logger.experiment["lyrics/train_gen"].log(gen_lyrics_syllable[0])
                gen_lyrics_word = sequences_to_texts(gen_samples_word, tokenizer_lyr_path=self.hparams.tokenizer_word_path)
                self.logger.experiment["lyrics/train_gen"].log(gen_lyrics_word[0])

        loss_syllable = self._cal_loss(pred_syllable, gold_syllable, label_smoothing=self.hparams.label_smoothing)
        loss_word = self._cal_loss(pred_word, gold_word)

        # padding_mask = self._get_pad_mask(seq_pitch[:, 1:-1], self.hparams.pad_idx_pitch)
        # distance_penalty = self._cal_regularization(z[:, 1:, :], feature_coordinates.to(device), padding_mask=padding_mask)
        if self.hparams.n_gram_alpha > 0:
            loss_ngram_syllable = self._cal_explicit_ngram_loss(pred_syllable, gold_syllable, position_flags[:, 1:],
                                                                padding_idx=self.hparams.pad_idx_syllable)
        else:
            loss_ngram_syllable = 0

        if self.hparams.multi_level:
            loss = loss_syllable + self.hparams.n_gram_alpha * loss_ngram_syllable + loss_word
        else:
            loss = loss_syllable + self.hparams.n_gram_alpha * loss_ngram_syllable

        self.log("loss_syllable_train", loss_syllable)
        self.log("loss_word_train", loss_word)
        # self.log("loss_distance_penalty", distance_penalty)
        self.log("loss_ngram_syllable", loss_ngram_syllable)
        # self.log("loss_ngram_word", loss_ngram_word)

        acc = self._cal_accuracy(torch.argmax(gen_samples_syllable, dim=2), gold_syllable, self.hparams.pad_idx_syllable)
        self.log("acc_train", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        seq_pitch, seq_duration, seq_rest = batch[0]
        target_syllable, target_word = batch[1]
        position_flags = batch[2]
        bs = target_syllable.shape[0]

        if self.hparams.prior_attn:
            position_flags = position_flags[:, 1:]
            # position_flags = position_flags[:, :-1]

            prior_attention_masks = self._generate_block_diagonal(position_flags,
                                                                  diagonal_fill=self.hparams.diagonal_fill).to(device)
            # torch.set_printoptions(profile="full")
            # print(position_flags[0])
            # print(prior_attention_masks[0])
            translator = TranslatorPrior(
                model=self.generator,
                beam_size=3,
                max_seq_len=self.hparams.max_seq_len + 1,  # bos + eos
                src_pad_idx=[self.hparams.pad_idx_pitch, self.hparams.pad_idx_duration, self.hparams.pad_idx_rest],
                trg_pad_idx=self.hparams.pad_idx_syllable,
                trg_bos_idx=self.hparams.bos_idx_syllable,
                trg_eos_idx=self.hparams.pad_idx_syllable).to(device)

            pred_list = []
            gold_list = []
            for p, d, r, lrc, attn in zip(seq_pitch, seq_duration, seq_rest, target_syllable, prior_attention_masks):
                src_seq_pitch = p[:-1].unsqueeze(0)
                src_seq_duration = d[:-1].unsqueeze(0)
                src_seq_rest = r[:-1].unsqueeze(0)
                trg_seq = lrc[:-1].unsqueeze(0)
                gold_seq = lrc[1:].unsqueeze(0)
                prior_attn = attn.unsqueeze(0)

                src_seq = src_seq_pitch.to(device), src_seq_duration.to(device), src_seq_rest.to(device)

                pred_seq = translator.translate_sentence(src_seq, prior_attn=prior_attn)
                pred_seq = torch.LongTensor(pred_seq[1:]).unsqueeze(0)
                # here, pad = (padding_left, padding_right, padding_top, padding_bottom)
                pred_seq = F.pad(pred_seq, pad=(0, gold_seq.shape[1] - pred_seq.shape[1], 0, 0),
                                 mode='constant', value=self.hparams.pad_idx_syllable).to(device)

                assert pred_seq.shape == gold_seq.shape

                pred_list.append(pred_seq)
                gold_list.append(gold_seq)
        else:
            translator = Translator(
                model=self.generator,
                beam_size=3,
                max_seq_len=self.hparams.max_seq_len + 1, # bos + eos
                src_pad_idx=[self.hparams.pad_idx_pitch, self.hparams.pad_idx_duration, self.hparams.pad_idx_rest],
                trg_pad_idx=self.hparams.pad_idx_syllable,
                trg_bos_idx=self.hparams.bos_idx_syllable,
                trg_eos_idx=self.hparams.pad_idx_syllable).to(device)

            pred_list = []
            gold_list = []
            for p, d, r, lrc in zip(seq_pitch, seq_duration, seq_rest, target_syllable):
                src_seq_pitch = p[:-1].unsqueeze(0)
                src_seq_duration = d[:-1].unsqueeze(0)
                src_seq_rest = r[:-1].unsqueeze(0)
                trg_seq = lrc[:-1].unsqueeze(0)
                gold_seq = lrc[1:].unsqueeze(0)

                src_seq = src_seq_pitch.to(device), src_seq_duration.to(device), src_seq_rest.to(device)

                pred_seq = translator.translate_sentence(src_seq)
                pred_seq = torch.LongTensor(pred_seq[1:]).unsqueeze(0)
                # here, pad = (padding_left, padding_right, padding_top, padding_bottom)
                pred_seq = F.pad(pred_seq, pad=(0, gold_seq.shape[1] - pred_seq.shape[1], 0, 0),
                                 mode='constant', value=self.hparams.pad_idx_syllable).to(device)

                assert pred_seq.shape == gold_seq.shape

                pred_list.append(pred_seq)
                gold_list.append(gold_seq)

        pred = torch.vstack(pred_list).to(device)
        gold = torch.vstack(gold_list).to(device)

        pred_np = pred[0].detach().cpu().numpy()
        gold_np = gold[0].detach().cpu().numpy()

        print('pred: ', pred_np)
        print('gold: ', gold_np)

        # gen_samples = F.softmax(pred, dim=-1)
        gen_samples = F.one_hot(pred, self.hparams.syllable_vocab_size).float()
        real_samples = F.one_hot(gold, self.hparams.syllable_vocab_size).float()

        gen_lyrics = sequences_to_texts(gen_samples, tokenizer_lyr_path=self.hparams.tokenizer_syllable_path)
        gt_lyrics = sequences_to_texts(real_samples, tokenizer_lyr_path=self.hparams.tokenizer_syllable_path)

        if batch_idx == 0:
            if self.current_epoch == 0:
                self.logger.experiment["lyrics/valid_gt"].log(gt_lyrics[0])

            # if (self.current_epoch + 1) % 20 == 0:
            if 1:
                self.logger.experiment["lyrics/valid_gen"].log(gen_lyrics[0])

        gen_lyrics = remove_start_and_end(gen_lyrics)
        gt_lyrics = remove_start_and_end(gt_lyrics)

        rouge_1, rouge_2, rouge_l = get_rouge_scores(gt_lyrics, gen_lyrics)
        self.log("val/rouge_1/precision", rouge_1[0], on_epoch=True)
        self.log("val/rouge_1/recall", rouge_1[1], on_epoch=True)
        self.log("val/rouge_1/f1", rouge_1[2], on_epoch=True)
        self.log("val/rouge_2/precision", rouge_2[0], on_epoch=True)
        self.log("val/rouge_2/recall", rouge_2[1], on_epoch=True)
        self.log("val/rouge_2/f1", rouge_2[2], on_epoch=True)
        self.log("val/rouge_l/precision", rouge_l[0], on_epoch=True)
        self.log("val/rouge_l/recall", rouge_l[1], on_epoch=True)
        self.log("val/rouge_l/f1", rouge_l[2], on_epoch=True)

        bleu_2, bleu_3, bleu_4 = get_bleu_scores(gt_lyrics, gen_lyrics)
        self.log("val/bleu/bleu_2", bleu_2, on_epoch=True)
        self.log("val/bleu/bleu_3", bleu_3, on_epoch=True)
        self.log("val/bleu/bleu_4", bleu_4, on_epoch=True)

        acc = self._cal_accuracy(pred, gold, self.hparams.pad_idx_syllable)
        self.log("acc_valid", acc, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None, usr_attn=True):
        seq_pitch, seq_duration, seq_rest = batch[0]
        target_syllable, target_word = batch[1]
        position_flags = batch[2]
        bs = target_syllable.shape[0]

        if self.hparams.prior_attn:
            position_flags = position_flags[:, 1:]
            # position_flags = position_flags[:, :-1]

            if usr_attn:
                position_flags[:, :] = 0
                position_flags[:, 1] = 1
                position_flags[:, 2] = 3
                position_flags[:, 3] = 3
                position_flags[:, 4] = 2
                # position_flags[:, 5] = 2

            prior_attention_masks = self._generate_block_diagonal(position_flags,
                                                                  diagonal_fill=self.hparams.diagonal_fill).to(device)

            # torch.set_printoptions(profile="full")
            # print(position_flags[0])
            # print(prior_attention_masks[0])
            translator = TranslatorPrior(
                model=self.generator,
                beam_size=3,
                max_seq_len=self.hparams.max_seq_len + 1,  # bos + eos
                src_pad_idx=[self.hparams.pad_idx_pitch, self.hparams.pad_idx_duration, self.hparams.pad_idx_rest],
                trg_pad_idx=self.hparams.pad_idx_syllable,
                trg_bos_idx=self.hparams.bos_idx_syllable,
                trg_eos_idx=self.hparams.pad_idx_syllable).to(device)

            pred_list = []
            gold_list = []
            for p, d, r, lrc, attn in zip(seq_pitch, seq_duration, seq_rest, target_syllable, prior_attention_masks):
                src_seq_pitch = p[:-1].unsqueeze(0)
                src_seq_duration = d[:-1].unsqueeze(0)
                src_seq_rest = r[:-1].unsqueeze(0)
                trg_seq = lrc[:-1].unsqueeze(0)
                gold_seq = lrc[1:].unsqueeze(0)
                prior_attn = attn.unsqueeze(0)

                src_seq = src_seq_pitch.to(device), src_seq_duration.to(device), src_seq_rest.to(device)

                pred_seq = translator.translate_sentence(src_seq, prior_attn=prior_attn)
                pred_seq = torch.LongTensor(pred_seq[1:]).unsqueeze(0)
                # here, pad = (padding_left, padding_right, padding_top, padding_bottom)
                pred_seq = F.pad(pred_seq, pad=(0, gold_seq.shape[1] - pred_seq.shape[1], 0, 0),
                                 mode='constant', value=self.hparams.pad_idx_syllable).to(device)

                assert pred_seq.shape == gold_seq.shape

                pred_list.append(pred_seq)
                gold_list.append(gold_seq)
        else:
            translator = Translator(
                model=self.generator,
                beam_size=3,
                max_seq_len=self.hparams.max_seq_len + 1, # bos + eos
                src_pad_idx=[self.hparams.pad_idx_pitch, self.hparams.pad_idx_duration, self.hparams.pad_idx_rest],
                trg_pad_idx=self.hparams.pad_idx_syllable,
                trg_bos_idx=self.hparams.bos_idx_syllable,
                trg_eos_idx=self.hparams.pad_idx_syllable).to(device)

            pred_list = []
            gold_list = []
            for p, d, r, lrc in zip(seq_pitch, seq_duration, seq_rest, target_syllable):
                src_seq_pitch = p[:-1].unsqueeze(0)
                src_seq_duration = d[:-1].unsqueeze(0)
                src_seq_rest = r[:-1].unsqueeze(0)
                trg_seq = lrc[:-1].unsqueeze(0)
                gold_seq = lrc[1:].unsqueeze(0)

                src_seq = src_seq_pitch.to(device), src_seq_duration.to(device), src_seq_rest.to(device)

                seq_len = self.hparams.max_seq_len

                pred_seq = translator.translate_sentence(src_seq)
                pred_seq = torch.LongTensor(pred_seq[1:]).unsqueeze(0)
                # here, pad = (padding_left, padding_right, padding_top, padding_bottom)
                pred_seq = F.pad(pred_seq, pad=(0, gold_seq.shape[1] - pred_seq.shape[1], 0, 0),
                                 mode='constant', value=self.hparams.pad_idx_syllable).to(device)

                assert pred_seq.shape == gold_seq.shape

                pred_list.append(pred_seq)
                gold_list.append(gold_seq)

        pred = torch.vstack(pred_list).to(device)
        gold = torch.vstack(gold_list).to(device)

        gen_samples = F.one_hot(pred, self.hparams.syllable_vocab_size).float()
        real_samples = F.one_hot(gold, self.hparams.syllable_vocab_size).float()

        gen_lyrics = sequences_to_texts(gen_samples, tokenizer_lyr_path=self.hparams.tokenizer_syllable_path)
        gt_lyrics = sequences_to_texts(real_samples, tokenizer_lyr_path=self.hparams.tokenizer_syllable_path)

        gen_lyrics = remove_start_and_end(gen_lyrics)
        gt_lyrics = remove_start_and_end(gt_lyrics)

        return gen_lyrics, gt_lyrics


    def generate(self, melody, position_flags, gt=None):
        if len(melody.shape) == 2 and len(position_flags.shape) == 1:
            melody = np.expand_dims(melody, 0)
            position_flags = np.expand_dims(position_flags, 0)

        seq_pitch = torch.LongTensor(melody[:, :, 0])
        seq_duration = torch.LongTensor(melody[:, :, 1])
        seq_rest = torch.LongTensor(melody[:, :, 2])
        position_flags = torch.LongTensor(position_flags)

        if self.hparams.prior_attn:
            position_flags = position_flags[:, 1:]
            # position_flags = position_flags[:, :-1]

            prior_attention_masks = self._generate_block_diagonal(position_flags,
                                                                  diagonal_fill=self.hparams.diagonal_fill).to(device)

            translator = TranslatorPrior(
                model=self.generator,
                beam_size=3,
                max_seq_len=self.hparams.max_seq_len + 1,  # bos + eos
                src_pad_idx=[self.hparams.pad_idx_pitch, self.hparams.pad_idx_duration, self.hparams.pad_idx_rest],
                trg_pad_idx=self.hparams.pad_idx_syllable,
                trg_bos_idx=self.hparams.bos_idx_syllable,
                trg_eos_idx=self.hparams.pad_idx_syllable).to(device)

            pred_list = []
            gold_list = []
            for p, d, r, attn in zip(seq_pitch, seq_duration, seq_rest, prior_attention_masks):
                src_seq_pitch = p[:-1].unsqueeze(0)
                src_seq_duration = d[:-1].unsqueeze(0)
                src_seq_rest = r[:-1].unsqueeze(0)
                prior_attn = attn.unsqueeze(0)

                src_seq = src_seq_pitch.to(device), src_seq_duration.to(device), src_seq_rest.to(device)

                pred_seq = translator.translate_sentence(src_seq, prior_attn=prior_attn)
                pred_seq = torch.LongTensor(pred_seq[1:]).unsqueeze(0)

                pred_list.append(pred_seq)
        else:
            translator = Translator(
                model=self.generator,
                beam_size=3,
                max_seq_len=self.hparams.max_seq_len + 1, # bos + eos
                src_pad_idx=[self.hparams.pad_idx_pitch, self.hparams.pad_idx_duration, self.hparams.pad_idx_rest],
                trg_pad_idx=self.hparams.pad_idx_syllable,
                trg_bos_idx=self.hparams.bos_idx_syllable,
                trg_eos_idx=self.hparams.pad_idx_syllable).to(device)

            pred_list = []
            for p, d, r in zip(seq_pitch, seq_duration, seq_rest):
                src_seq_pitch = p[:-1].unsqueeze(0)
                src_seq_duration = d[:-1].unsqueeze(0)
                src_seq_rest = r[:-1].unsqueeze(0)

                src_seq = src_seq_pitch.to(device), src_seq_duration.to(device), src_seq_rest.to(device)

                pred_seq = translator.translate_sentence(src_seq)
                pred_seq = torch.LongTensor(pred_seq[1:]).unsqueeze(0)
                # here, pad = (padding_left, padding_right, padding_top, padding_bottom)

                pred_list.append(pred_seq)

        pred = torch.vstack(pred_list).to(device)

        gen_samples = F.one_hot(pred, self.hparams.syllable_vocab_size).float()

        gen_lyrics = sequences_to_texts(gen_samples, tokenizer_lyr_path=self.hparams.tokenizer_syllable_path)

        gen_lyrics = remove_start_and_end(gen_lyrics)

        return gen_lyrics
