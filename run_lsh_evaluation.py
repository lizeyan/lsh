import click
from lsh import LSH, BasicE2LSH
from pathlib import Path
import numpy as np
import seaborn as sns
from loguru import logger


data_base_path = Path("./outputs")
train_data = np.memmap(data_base_path / 'train_arr', mode='r', dtype=np.float32)
test_data = np.memmap(data_base_path / 'test_arr', mode='r', dtype=np.float32)
ground_truth = np.memmap(data_base_path / 'ground_truth', mode='r', dtype=np.float32)


def lsh_evaluation(name, lsh: LSH):
    logger.info(f"start add entries, train data shape: {np.shape(train_data)}")
    for q in train_data:
        lsh.add(q)
    logger.info(f"evaluate")


@click.command()
def main():
    lsh_evaluation("BasicE2LSH", BasicE2LSH(n_dims=50, n_hash_table=20, n_compounds=20, w=4.))


if __name__ == '__main__':
    main()
