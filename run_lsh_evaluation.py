from pathlib import Path

import click
import numpy as np
from loguru import logger

from config import n_dims, n_test_samples, n_ground_truth, n_train_samples
from lsh import LSH, BasicE2LSH


data_base_path = Path("./outputs")
train_data = np.memmap(data_base_path / 'train_arr', mode='r', dtype=np.float32, shape=(n_train_samples, n_dims))
test_data = np.memmap(data_base_path / 'test_arr', mode='r', dtype=np.float32, shape=(n_test_samples, n_dims))
ground_truth = np.memmap(
    data_base_path / 'ground_truth', mode='r', dtype=np.int32, shape=(n_test_samples, n_ground_truth)
)


def lsh_evaluation(name, lsh: LSH):
    logger.info(f"{name}")
    logger.info(f"start add entries, train data shape: {np.shape(train_data)}")
    lsh.add_batch(train_data[:60000])
    logger.info(f"start evaluate")
    for idx, test_sample in enumerate(test_data):
        candidates = lsh.query(test_sample)
        print(len(candidates))



@click.command()
def main():
    lsh_evaluation("BasicE2LSH", BasicE2LSH(n_dims=50, n_hash_table=20, n_compounds=20, w=10.))


if __name__ == '__main__':
    main()
