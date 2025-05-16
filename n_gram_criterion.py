"""
@Project: controllable-syllable-level-lyrics-generation-from-melody-with-prior-attention
@File: n_gram_loss.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
"""
# Implementation of the Explicit N-Gram (EXPLING) loss for syllable-level
# lyrics generation, capturing structural dependencies of multi-syllable words.
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.Layers import EncoderLayer, DecoderLayer
from utils import *
from typing import Union, Tuple, Optional


class ExplicitNGramCriterion(nn.Module):
    """
    EXPLING loss module computes a masked negative log-likelihood over
    syllable positions belonging to multi-syllable words, encouraging the
    model to capture correct syllable groupings within words.
    """
    def __init__(
            self,):

        super().__init__()

    # Build a binary mask over time steps where flags indicate syllable spans:
    # flag==1 marks the beginning, flag==2 marks the end of words.
    def config_nll_mask(self, flags: torch.Tensor, return_count: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor,int]]:
        mask = torch.zeros_like(flags)
        begin_flags = (flags == 1).nonzero(as_tuple=False)
        end_flags = (flags == 2).nonzero(as_tuple=False)

        for begin, end in zip(begin_flags, end_flags):
            assert begin[0] == end[0] and begin[1] < end[1], "begin: {}, end: {}".format(begin, end)
            mask[begin[0], begin[1]:end[1] + 1] = 1

        if return_count:
            return mask, len(begin_flags)
        else:
            return mask

    # Compute masked NLL: gather log-probabilities of target indices,
    # apply mask to select multi-syllable positions, normalize by word count.
    def batch_syllable_nll(self, log_probs: torch.Tensor, target_idx: torch.Tensor, mask: torch.Tensor, word_count: int) -> torch.Tensor:
        batch_size, output_len, vocab_size = log_probs.size()
        _, tgt_len = target_idx.size()


        # [batch, output_len, target_len]
        index = target_idx.unsqueeze(-1)
        log_probs_gathered = log_probs.gather(dim=-1, index=index)
        log_probs_gathered = log_probs_gathered.squeeze(-1)

        log_probs_masked = log_probs_gathered.mul(mask)

        term = log_probs_masked.sum() / word_count

        # loss = - term / batch_size
        loss = - term

        return loss

    # Forward pass: compute log-softmax of logits, construct mask and word count,
    # then compute EXPLING loss over the batch.
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, flags: torch.Tensor, padding_idx: Optional[int] = None) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        mask, word_count = self.config_nll_mask(flags, return_count=True)
        loss = self.batch_syllable_nll(log_probs, targets, mask, word_count)

        return loss
