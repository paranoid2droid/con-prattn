"""
@Project: controllable-syllable-level-lyrics-generation-from-melody-with-prior-attention
@File: evaluation.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
"""

# This module provides evaluation metrics for melody-to-lyrics generation,
# including ROUGE, BLEU, ChrF, corpus BLEU, BERTScore, and InfoLM.

# Standard library imports
import numpy as np
from tqdm import tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Metric libraries imports
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bert import BERTScore
from torchmetrics.functional import chrf_score
from torchmetrics.functional.text.bert import bert_score
from torchmetrics.text.infolm import InfoLM

# Number of decimal places to round metric outputs.
NUM_DECIAML = 3

# Compute ROUGE-1, ROUGE-2, and ROUGE-L scores (precision, recall, F1)
# between reference and predicted lyrics lists.
def get_rouge_scores(orig, preds):

    rouge = ROUGEScore()

    r_f_measure_1, r_precision_1, r_recall_1 = [], [], []
    r_f_measure_2, r_precision_2, r_recall_2 = [], [], []
    r_f_measure_l, r_precision_l, r_recall_l = [], [], []
    for test_ref, test_pred in zip(orig, preds):
        rouge_dict = rouge(test_pred, test_ref)
        rouge1_fmeasure = rouge_dict["rouge1_fmeasure"]
        rouge1_precision = rouge_dict["rouge1_precision"]
        rouge1_recall = rouge_dict["rouge1_recall"]
        rouge2_fmeasure = rouge_dict["rouge2_fmeasure"]
        rouge2_precision = rouge_dict["rouge2_precision"]
        rouge2_recall = rouge_dict["rouge2_recall"]
        rougeL_fmeasure = rouge_dict["rougeL_fmeasure"]
        rougeL_precision = rouge_dict["rougeL_precision"]
        rougeL_recall = rouge_dict["rougeL_recall"]

        r_f_measure_1.append(rouge1_fmeasure)
        r_precision_1.append(rouge1_precision)
        r_recall_1.append(rouge1_recall)
        r_f_measure_2.append(rouge2_fmeasure)
        r_precision_2.append(rouge2_precision)
        r_recall_2.append(rouge2_recall)
        r_f_measure_l.append(rougeL_fmeasure)
        r_precision_l.append(rougeL_precision)
        r_recall_l.append(rougeL_recall)

    # print(np.mean(r_f_measure_1), np.mean(r_precision_1), np.mean(r_recall_1))
    # print(np.mean(r_f_measure_2), np.mean(r_precision_2), np.mean(r_recall_2))
    # print(np.mean(r_f_measure_l), np.mean(r_precision_l), np.mean(r_recall_l))

    rouge_1_f1 = np.mean(r_f_measure_1)
    rouge_1_precision = np.mean(r_precision_1)
    rouge_1_recall = np.mean(r_recall_1)

    rouge_2_f1 = np.mean(r_f_measure_2)
    rouge_2_precision = np.mean(r_precision_2)
    rouge_2_recall = np.mean(r_recall_2)

    rouge_l_f1 = np.mean(r_f_measure_l)
    rouge_l_precision = np.mean(r_precision_l)
    rouge_l_recall = np.mean(r_recall_l)

    return [rouge_1_precision.round(NUM_DECIAML), rouge_1_recall.round(NUM_DECIAML), rouge_1_f1.round(NUM_DECIAML)], \
           [rouge_2_precision.round(NUM_DECIAML), rouge_2_recall.round(NUM_DECIAML), rouge_2_f1.round(NUM_DECIAML)], \
           [rouge_l_precision.round(NUM_DECIAML), rouge_l_recall.round(NUM_DECIAML), rouge_l_f1.round(NUM_DECIAML)]


# Compute sentence-level BLEU-2, BLEU-3, and BLEU-4 scores with smoothing.
def get_bleu_scores(orig, preds):
    chencherry = SmoothingFunction()

    bleus_4, bleus_3, bleus_2 = [], [], []
    for test_ref, test_pred in zip(orig, preds):
        test_ref = test_ref.split(' ')
        test_pred = test_pred.split(' ')
        bleu4 = sentence_bleu([test_ref], test_pred, smoothing_function=chencherry.method1)
        bleu3 = sentence_bleu([test_ref], test_pred, weights=[1 / 3, 1 / 3, 1 / 3], smoothing_function=chencherry.method1)
        bleu2 = sentence_bleu([test_ref], test_pred, weights=[1 / 2, 1 / 2], smoothing_function=chencherry.method1)
        bleus_4.append(bleu4)
        bleus_3.append(bleu3)
        bleus_2.append(bleu2)

    return np.mean(bleus_2).round(NUM_DECIAML), np.mean(bleus_3).round(NUM_DECIAML), np.mean(bleus_4).round(NUM_DECIAML)


# Compute character F-score (ChrF) and ChrF++ for each prediction.
def get_chrf_scores(orig, preds):
    chrf_list, chuf_plus_list = [], []
    for test_target, test_pred in zip(orig, preds):
        chrf = chrf_score(test_pred, [test_target], n_word_order=0)
        chrf_plus = chrf_score(test_pred, [test_target], n_word_order=2)
        chrf_list.append(chrf)
        chuf_plus_list.append(chrf_plus)

    return np.mean(chrf_list).round(NUM_DECIAML), np.mean(chuf_plus_list).round(NUM_DECIAML)


# Compute corpus-level BLEU scores for n-gram orders 2 to 5.
def get_corpus_bleu_scores(orig, preds):
    # >>> list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
    # >>> hypotheses = [hyp1, hyp2]

    list_of_references = []
    hypotheses = []

    for test_ref, test_pred in zip(orig, preds):
        test_ref = test_ref.split(' ')
        test_pred = test_pred.split(' ')
        list_of_references.append([test_ref])
        hypotheses.append(test_pred)

    weights = [
        (0.5, 0.5),
        (0.333, 0.333, 0.334),
        (0.25, 0.25, 0.25, 0.25),
        (0.2, 0.2, 0.2, 0.2, 0.2)
    ]
    bleu_corpus = corpus_bleu(list_of_references, hypotheses, weights=weights)

    return list(np.around(np.array(bleu_corpus), NUM_DECIAML))


# Compute BERTScore (precision, recall, F1) between predicted and reference texts.
def get_bert_scores(orig, preds):
    bertscore = BERTScore(model_name_or_path="microsoft/deberta-xlarge-mnli", idf=False)

    # bert_precision_list = []
    # bert_recall_list = []
    # bert_f1_list = []
    #
    # for test_target, test_pred in tqdm(zip(orig, preds), total=len(orig)):
    #     bert_score = bertscore(test_pred, test_target)
    #     bert_precision_list.append(bert_score["precision"])
    #     bert_recall_list.append(bert_score["recall"])
    #     bert_f1_list.append(bert_score["f1"])
    #
    # return [np.mean(bert_precision_list).round(NUM_DECIAML), np.mean(bert_recall_list).round(NUM_DECIAML), np.mean(bert_f1_list).round(NUM_DECIAML)]

    scores = bertscore(preds, orig)
    return [np.mean(scores["precision"]).round(NUM_DECIAML), np.mean(scores["recall"]).round(NUM_DECIAML), np.mean(scores["f1"]).round(NUM_DECIAML)]


# Compute InfoLM KL divergence and L2 distance metrics for predicted vs reference.
def get_infolm_scores(orig, preds):
    infolm_kl = InfoLM('google/bert_uncased_L-2_H-128_A-2', idf=False, information_measure='alpha_divergence', alpha=0.5,
                       verbose=False)
    infolv_l2 = InfoLM('google/bert_uncased_L-2_H-128_A-2', idf=False, information_measure='fisher_rao_distance',
                       verbose=False)

    kl_divergence_list, l2_distance_list = [], []
    for test_target, test_pred in tqdm(zip(orig, preds), total=len(orig)):
        kl_divergence = infolm_kl(test_pred, test_target)
        l2_distance = infolv_l2(test_pred, test_target)
        # print(kl_divergence)
        # print(l2_distance)
        kl_divergence_list.append(kl_divergence.numpy())
        l2_distance_list.append(l2_distance.numpy())

    return np.mean(kl_divergence_list).round(NUM_DECIAML), np.mean(l2_distance_list).round(NUM_DECIAML)
