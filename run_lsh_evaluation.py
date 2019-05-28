from pathlib import Path

import click
import numpy as np
from loguru import logger
import time
from config import n_dims, n_test_samples, n_train_samples
from lsh import LSH, BasicE2LSH
from lsh.multi_probe_lsh import MultiProbeE2LSH

data_base_path = Path("./outputs")
train_data = np.memmap(data_base_path / 'train_arr', mode='r', dtype=np.float32, shape=(n_train_samples, n_dims))
test_data = np.memmap(data_base_path / 'test_arr', mode='r', dtype=np.float32, shape=(n_test_samples, n_dims))
ground_truth = np.memmap(
    data_base_path / 'ground_truth', mode='r', dtype=np.int32, shape=(n_test_samples, n_train_samples)
)


def e2_ratio(q, labels, preds, max_k=20):
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


def lsh_evaluation(lsh: LSH, **kwargs):
    handler_id = logger.add(data_base_path / 'logs' / f'{lsh}.log'.replace(' ', '_'))
    logger.info(lsh)
    logger.info(f"start add entries, train data shape: {np.shape(train_data)}")
    tic = time.time()
    lsh.add_batch(train_data[:60000])
    toc = time.time()
    logger.info(f"build hash table cost: {toc - tic}s")
    logger.info(f"start evaluate")
    error_ratio_list = []
    for idx, test_sample in enumerate(test_data):
        candidates = lsh.query(test_sample, **kwargs)
        n_candidates = np.size(candidates, 0)
        if n_candidates <= 0:
            continue
        labels = train_data[ground_truth[idx, :n_candidates]]
        error_ratio_list.append(e2_ratio(test_sample, labels, candidates))
    logger.info(f"error ratio: {np.mean(error_ratio_list)}")
    logger.remove(handler_id)


@click.command()
def main():
    lsh_evaluation(BasicE2LSH(n_dims=50, n_hash_table=10, n_compounds=20, w=10.))
    lsh_evaluation(MultiProbeE2LSH(n_dims=50, n_hash_table=5, n_compounds=20, w=10.), t=100)


if __name__ == '__main__':
    main()
