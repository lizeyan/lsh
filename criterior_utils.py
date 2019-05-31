import json
import sys
from pathlib import Path

import click
import numpy as np
from loguru import logger

from config import n_dims, n_test_samples, n_train_samples
from lsh import LSH, BasicE2LSH
from utility import Timer

data_base_path = Path("./outputs")
train_data = np.memmap(data_base_path / 'train_arr', mode='r', dtype=np.float32, shape=(n_train_samples, n_dims))
test_data = np.memmap(data_base_path / 'test_arr', mode='r', dtype=np.float32, shape=(n_test_samples, n_dims))
ground_truth = np.memmap(
    data_base_path / 'ground_truth', mode='r', dtype=np.int32, shape=(n_test_samples, n_train_samples)
)


def e2(x, y):
    return np.sqrt(np.sum((x - y) ** 2, axis=-1))


def get_e2_ratio(q, labels, preds, preds_dis, max_k=20):
    """
    :param preds_dis: preds's distance to q
    :param max_k: calc min(preds, max_k) for q
    :param q: the query vector in shape (d,)
    :param labels: label, in shape (k, d)
    :param preds: predict, in shape (k, d)
    :return:
    """
    q = np.expand_dims(q.copy(), 0)
    length = min(np.size(preds, 0), max_k)
    ratio_list = preds_dis[:length] / e2(q, labels[:length])
    assert np.nanmax(ratio_list) >= 1.0
    return np.nanmean(ratio_list)


def get_precision_recall(q, labels, preds, max_k=20):
    """
    :param max_k: consider how many labels as ground truth
    :param q: the query vector in shape (d,)
    :param labels: label, in shape (k, d)
    :param preds: predict, in shape (k, d) in sorted order
    :return:
    """
    intersection = (set(map(tuple, preds)).intersection(set(map(tuple, labels[:max_k]))))
    return len(intersection) / len(preds), len(intersection) / min(max_k, len(labels))


def lsh_evaluation(lsh: LSH, **kwargs):
    max_k = 10
    with Timer() as total_timer:
        name = "&".join([str(lsh)] + [f'{key}={value}' for key, value in kwargs.items()])
        handler_id = logger.add(data_base_path / 'logs' / f'{name}.log')
        logger.info(name)
        logger.info(f"start add entries, train data shape: {np.shape(train_data)}")
        with Timer() as build_timer:
            lsh.add_batch(train_data[:60000])
            #lsh.add_batch(test_data)
        logger.info(f"build hash table cost: {build_timer.elapsed_time:.4f}s")
        logger.info(f"start evaluate")
        with Timer() as search_timer:
            error_ratio_list = []
            recall_list = []
            precision_list = []
            cn_list = []
            for idx, test_sample in enumerate(test_data):
                candidates = lsh.query(test_sample, **kwargs)
                if not len(candidates):
                    cn_list.append(0)
                    precision_list.append(float('nan'))
                    recall_list.append(0)
                    error_ratio_list.append(float('nan'))
                else:
                    preds_dis = e2(test_sample, candidates)
                    indices = np.argsort(preds_dis)
                    preds_dis = preds_dis[indices]
                    candidates = candidates[indices]
                    labels = train_data[ground_truth[idx, :]]
                    error_ratio_list.append(get_e2_ratio(test_sample, labels, candidates, preds_dis, max_k=max_k))
                    _p, _r = get_precision_recall(test_sample, labels, candidates, max_k=max_k)
                    recall_list.append(_r)
                    precision_list.append(_p)
                    cn_list.append(len(candidates) / len(train_data))
            error_ratio = np.nanmean(error_ratio_list)
            recall = np.nanmean(recall_list)
            precision = np.nanmean(precision_list)
            cn = np.nanmean(cn_list)
    ret = {
        'total_time': total_timer.elapsed_time,
        'error_ratio': error_ratio,
        'build_time': build_timer.elapsed_time,
        'search_time': search_timer.elapsed_time,
        'recall': recall,
        'precision': precision,
        'c/n': cn,
    }
    logger.info(ret)
    logger.remove(handler_id)
    return ret


