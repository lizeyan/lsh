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
from criterior_utils import lsh_evaluation, train_data
 




@click.command()
@click.option('--n-hash-table', default=4)
@click.option('--n-compounds', default=4)
@click.option('--w', default=1.0)
def main(n_hash_table, n_compounds, w):
    #LSHForest(l=n_hash_table, km=n_compounds).add_batch(train_data)
    ret = lsh_evaluation(LSHForest(l=n_hash_table, km=n_compounds))
    ret['n_hash_table'] = n_hash_table
    ret['n_compounds'] = n_compounds
    ret['w'] = w
    ret['algorithm'] = 'LSHForest'
    print(ret)

if __name__ == '__main__':
    with Timer() as main_experiment_timer:
        main()
    logger.info(f"experiment finished, cost {main_experiment_timer.elapsed_time}")
