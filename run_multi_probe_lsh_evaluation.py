import copy
from itertools import product
from pathlib import Path
import pandas as pd
import click
import numpy as np
from loguru import logger
import time
from config import n_dims, n_test_samples, n_train_samples
from lsh import LSH, BasicE2LSH
from lsh.multi_probe_lsh import MultiProbeE2LSH
from utility import Timer


data_base_path = Path("./outputs")
train_data = np.memmap(data_base_path / 'train_arr', mode='r', dtype=np.float32, shape=(n_train_samples, n_dims))
test_data = np.memmap(data_base_path / 'test_arr', mode='r', dtype=np.float32, shape=(n_test_samples, n_dims))
ground_truth = np.memmap(
    data_base_path / 'ground_truth', mode='r', dtype=np.int32, shape=(n_test_samples, n_train_samples)
)


def get_e2_ratio(q, labels, preds, max_k=20):
    """
    :param max_k: calc min(preds, max_k) for q
    :param q: the query vector in shape (d,)
    :param labels: label, in shape (k, d)
    :param preds: predict, in shape (k, d)
    :return:
    """
    e2 = lambda x, y: np.sqrt(np.sum((x - y) ** 2, axis=-1))
    q = np.expand_dims(q.copy(), 0)
    length = min(np.size(preds, 0), max_k)
    ratio_list = np.sort(e2(q, preds))[:length] / e2(q, labels[:length])
    assert np.max(ratio_list) >= 1.0
    return np.mean(ratio_list)


def get_recall(q, labels, preds, max_k=20):
    """
    :param max_k: consider how many labels as ground truth
    :param q: the query vector in shape (d,)
    :param labels: label, in shape (k, d)
    :param preds: predict, in shape (k, d)
    :return:
    """
    eps = 1e-2
    e2 = lambda x, y: np.sqrt(np.sum((x - y) ** 2, axis=-1))
    q = np.expand_dims(q.copy(), 0)
    preds_dis = e2(q, preds)
    labels_dis = e2(q, labels)
    preds = preds[np.argsort(preds_dis)]
    i = 0
    j = 0
    tp_count = 0
    while i < len(preds) and j < min(max_k, len(labels)):
        if preds_dis[i] < labels_dis[j] - eps:
            i += 1
        elif preds_dis[i] > labels_dis[j] + eps:
            j += 1
        else:
            i += 1
            j += 1
            tp_count += 1
    return tp_count / min(max_k, len(labels))


def lsh_evaluation(lsh: LSH, **kwargs):
    with Timer() as total_timer:
        handler_id = logger.add(data_base_path / 'logs' / f'{lsh}.log'.replace(' ', '_'))
        logger.info(lsh)
        logger.info(f"start add entries, train data shape: {np.shape(train_data)}")
        with Timer() as build_timer:
            lsh.add_batch(train_data[:60000])
        logger.info(f"build hash table cost: {build_timer.elapsed_time:.4f}s")
        logger.info(f"start evaluate")
        with Timer() as search_timer:
            error_ratio_list = []
            recall_list = []
            for idx, test_sample in enumerate(test_data):
                candidates = lsh.query(test_sample, **kwargs)
                if not len(candidates):
                    continue
                labels = train_data[ground_truth[idx, :]]
                error_ratio_list.append(get_e2_ratio(test_sample, labels, candidates))
                recall_list.append(get_recall(test_sample, labels, candidates, max_k=5))
            error_ratio = np.mean(error_ratio_list)
            recall = np.mean(recall_list)
    ret = {
        'total_time:': total_timer.elapsed_time,
        'error_ratio': error_ratio,
        'build_time': build_timer.elapsed_time,
        'search_time': search_timer.elapsed_time,
        'recall': recall,
        'algorithm': lsh.__class__.__name__,
    }
    logger.info(ret)
    logger.remove(handler_id)
    return ret


def main():
    result_list = []
    try:
        n_hash_table_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        n_compounds_list = [1, 2, 4, 8, 16, 32, 64]
        w_list = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        t_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        # n_hash_table_list = [1, 2, ]
        # n_compounds_list = [1, ]
        # w_list = [0.5, 1.0, ]
        # t_list = [1, 2, 4, ]
        result_list.extend(list(map(
            lambda params: lsh_evaluation(BasicE2LSH(
                n_dims=n_dims, n_hash_table=params[0], n_compounds=params[1], w=params[2])),
            product(n_hash_table_list, n_compounds_list, w_list)
        )))
        result_list.extend(list(map(
            lambda params: lsh_evaluation(MultiProbeE2LSH(
                n_dims=n_dims, n_hash_table=params[0], n_compounds=params[1], w=params[2]), t=params[3]),
            product(n_hash_table_list, n_compounds_list, w_list, t_list)
        )))
    except KeyboardInterrupt as e:
        logger.error(e)
    finally:
        result_df = pd.DataFrame.from_records(result_list)
        result_df.to_csv(data_base_path / 'multi_probe_lsh.csv', index=False)


if __name__ == '__main__':
    with Timer() as main_experiment_timer:
        main()
    logger.info(f"experiment finished, cost {main_experiment_timer.elapsed_time}")
