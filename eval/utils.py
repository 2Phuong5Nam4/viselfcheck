from ast import literal_eval

import numpy as np
import pandas as pd
from datasets import Dataset

from sklearn.metrics import precision_recall_curve, auc
from scipy.stats import pearsonr, spearmanr

from underthesea import text_normalize

def read_dataset(path):
    df = pd.read_csv(path, encoding='utf-8')

    for col in df.columns[2:]:
        df[col] = df[col].apply(lambda x: literal_eval(x))

    # eliminate \ufeff token
    df['gemini_text'] = df['gemini_text'].apply(lambda x: x.replace('\ufeff', ''))
    df['gemini_sentences'] = df['gemini_sentences'].apply(lambda x: [y.replace('\ufeff', '') for y in x])
    df['gemini_text_samples'] = df['gemini_text_samples'].apply(lambda x: [y.replace('\ufeff', '') for y in x])

    df['gemini_text'] = df['gemini_text'].apply(lambda x: text_normalize(x))
    df['gemini_sentences'] = df['gemini_sentences'].apply(lambda x: [text_normalize(y) for y in x])
    df['gemini_text_samples'] = df['gemini_text_samples'].apply(lambda x: [text_normalize(y) for y in x])

    return Dataset.from_pandas(df)

def label_to_score(label):
    label_mapping = {
        'accurate': 0.0,
        'minor_inaccurate': 0.5,
        'major_inaccurate': 1.0,
    }
    return label_mapping[label]

def get_scores_from_human_labels(dataset):
    human_label_detect_False   = {}
    human_label_detect_False_h = {}
    human_label_detect_True    = {}

    for idx, datapoint in enumerate(dataset):
        raw_label = np.array([label_to_score(x) for x in datapoint['annotation']])
        human_label_detect_False[idx] = (raw_label > 0.499).astype(np.int32).tolist()
        human_label_detect_True[idx]  = (raw_label < 0.499).astype(np.int32).tolist()
        average_score = np.mean(raw_label)

        if average_score < 0.99:
            human_label_detect_False_h[idx] = (raw_label > 0.99).astype(np.int32).tolist()

    return {
        'human_label_detect_False': human_label_detect_False,
        'human_label_detect_False_h': human_label_detect_False_h,
        'human_label_detect_True': human_label_detect_True
    }

def unroll_pred(scores, indices):
    unrolled = []
    for idx in indices:
        unrolled.extend(scores[idx])
    return unrolled

def evaluate(preds, human_labels, pos_label=1, oneminus_pred=False):
    indices = [k for k in human_labels.keys()]
    unroll_preds = unroll_pred(preds, indices)

    if oneminus_pred:
        unroll_preds = [1.0-x for x in unroll_preds]

    unroll_labels = unroll_pred(human_labels, indices)

    assert len(unroll_preds) == len(unroll_labels)

    # unroll_preds = np.clip(unroll_preds, -1e10, 1e10) # temporary fix
    P_, R_, thre_ = precision_recall_curve(unroll_labels, unroll_preds, pos_label=pos_label)
    auc_ = auc(R_, P_)

    return {
        'precision': P_,
        'recall': R_,
        'threshold': thre_,
        'auc': auc_
    }

def calc_corr(selfcheck_scores, dataset):
    selfcheck_scores_passages, label_passages = [], []
    for idx, datapoint in enumerate(dataset):
        selfcheck_scores_passages.append(np.mean(selfcheck_scores[idx]))
        label_passages.append(np.mean([label_to_score(x) for x in datapoint['annotation']]))

    pearsonr_ = pearsonr(selfcheck_scores_passages, label_passages).statistic
    spearmanr_ = spearmanr(selfcheck_scores_passages, label_passages).statistic

    return {
        'pearsonr': pearsonr_,
        'spearmanr': spearmanr_,
    }

def result_collect(selfcheck_scores, dataset, method, selfcheck_scores_passage_level=None):
    result = {}
    result['Method'] = method

    human_scores = get_scores_from_human_labels(dataset)
    result['NoFac'] = evaluate(selfcheck_scores, human_scores['human_label_detect_False'], pos_label=1)['auc']
    result['NoFac*'] = evaluate(selfcheck_scores, human_scores['human_label_detect_False_h'], pos_label=1)['auc']
    result['Fac'] = evaluate(selfcheck_scores, human_scores['human_label_detect_True'], pos_label=1, oneminus_pred=True)['auc']

    if selfcheck_scores_passage_level is None and 'gram' not in method.lower():
        corr_ = calc_corr(selfcheck_scores, dataset)
    else:
        corr_ = calc_corr(selfcheck_scores_passage_level, dataset)

    result['Pearson'] = corr_['pearsonr']
    result['Spearman'] = corr_['spearmanr']

    for k, v in result.items():
        if k != 'Method':
            result[k] = v * 100

    return result