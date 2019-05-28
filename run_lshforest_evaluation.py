from pathlib import Path

import click
import numpy as np
from loguru import logger
from LSH_forest.lsh_forest import LSHForest
from config import n_dims, n_test_samples, n_train_samples
from lsh import LSH, BasicE2LSH
from lsh.multi_probe_lsh import MultiProbeE2LSH
from utility import Timer
from concurrent.futures import ProcessPoolExecutor

data_base_path = Path("./outputs")
train_data = np.memmap(data_base_path / 'train_arr', mode='r', dtype=np.float32, shape=(n_train_samples, n_dims))
test_data = np.memmap(data_base_path / 'test_arr', mode='r', dtype=np.float32, shape=(n_test_samples, n_dims))
ground_truth = np.memmap(
    data_base_path / 'ground_truth', mode='r', dtype=np.int32, shape=(n_test_samples, n_train_samples)
)


def e2_ratio(q, labels, preds, max_k=100):
    """
    :param max_k: calc min(preds, max_k) for q
    :param q: the query vector in shape (d,)
    :param labels: label, in shape (k, d)
    :param preds: predict, in shape (k, d)
    :return:
    """
    eps = 1e-2
    e2 = lambda x, y: np.sqrt(np.sum((x - y) ** 2, axis=-1))
    q = np.expand_dims(q.copy(), 0)
    length = min(np.size(preds, 0), max_k)
    ratio_list = np.sort(e2(q, preds[:length])) / e2(q, labels[:length])
    return np.mean(ratio_list)


def lsh_evaluation(lsh: LSH, **kwargs):
    handler_id = logger.add(data_base_path / 'logs' / f'{lsh}.log'.replace(' ', '_'))
    logger.info(lsh)
    logger.info(f"start add entries, train data shape: {np.shape(train_data)}")
    lsh.add_batch(train_data[:60000])
    #lsh.add_batch(test_data)
    logger.info(f"start evaluate")
    error_ratio_list = []
    for idx, test_sample in enumerate(test_data):
        candidates = lsh.query(test_sample, **kwargs)
        n_candidates = np.size(candidates, 0)
        labels = train_data[ground_truth[idx, :n_candidates]]
        error_ratio_list.append(e2_ratio(test_sample, labels, candidates))
    #lsh.delete_one(train_data[0])
    logger.info(f"error ratio: {np.mean(error_ratio_list)}")
    logger.remove(handler_id)


@click.command()
def main():
    #lsh_evaluation(BasicE2LSH(n_dims=50, n_hash_table=44, n_compounds=10, w=10.))
    lsh_evaluation(LSHForest(l=40, km=30))
    # lsh_evaluation(MultiProbeE2LSH(n_dims=50, n_hash_table=44, n_compounds=10, w=10.), t=1)


if __name__ == '__main__':
    with Timer() as main_experiment_timer:
        main()
    logger.info(f"experiment finished, cost {main_experiment_timer.elapsed_time}")
