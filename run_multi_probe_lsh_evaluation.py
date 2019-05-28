import json

import click
from loguru import logger

from config import n_dims
from lsh.multi_probe_lsh import MultiProbeE2LSH
from run_basic_lsh_evaluation import lsh_evaluation
from utility import Timer


@click.command()
@click.option('--n-hash-table', default=4)
@click.option('--n-compounds', default=4)
@click.option('--w', default=1.0)
@click.option('--t', default=1)
def main(n_hash_table, n_compounds, w, t):
    ret = lsh_evaluation(MultiProbeE2LSH(n_dims=n_dims, n_hash_table=n_hash_table, n_compounds=n_compounds, w=w), t=t)
    ret['n_hash_table'] = n_hash_table
    ret['n_compounds'] = n_compounds
    ret['w'] = w
    ret['t'] = t
    ret['algorithm'] = 'MultiProbeLSH'
    print(json.dumps(ret))


if __name__ == '__main__':
    with Timer() as main_experiment_timer:
        main()
    logger.info(f"experiment finished, cost {main_experiment_timer.elapsed_time}")
