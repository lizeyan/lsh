import subprocess
import threading
from itertools import product
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime
import pandas as pd
from loguru import logger
import numpy as np
import random


base_path = '/home/lizytalk/Projects/lsh/'
server_list = [f'cpu{i}' for i in range(1, 11)]
server_avail = np.asarray([5 for _ in server_list])
lock = threading.Lock()


n_hash_table_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
n_compounds_list = [1, 2, 4, 8, 16, 32, 64]
w_list = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
t_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
# n_hash_table_list = [1, ]
# n_compounds_list = [20, ]
# w_list = [1, ]
# t_list = [10, ]


def find_avail_server():
    with lock:
        if np.max(server_avail) <= 0:
            return None
        else:
            idx = np.argmax(server_avail).item()
            server_avail[idx] -= 1
            return idx


def work(cmd):
    while True:
        server_idx = find_avail_server()
        if server_idx is not None:
            break
        else:
            time.sleep(5)
    try:
        server = server_list[server_idx]
        cmd = f"ssh {server} {cmd}"
        logger.debug(f"command: {cmd}")
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode()
        # logger.debug(output)
        ret = eval(output.replace('nan', 'None'))
        # time.sleep(30)
        # ret = {}
        logger.debug(f"ret: {ret}")
        logger.debug(f"release {server}")
        results.append(ret)
    except Exception as e:
        logger.error(f"error when executing {cmd}, {e}")
    finally:
        server_avail[server_idx] += 1


def worker_basic_lsh(params):
    n_hash_table, n_compounds, w = params
    cmd = f'\"cd {base_path}; source ~/.zshrc && ' \
        f'python3 {base_path}/run_basic_lsh_evaluation.py ' \
        f'--n-hash-table {n_hash_table} ' \
        f'--n-compounds {n_compounds} ' \
        f'--w {w} ' \
        f'\"'
    return work(cmd)


def worker_multi_probe_lsh(params):
    n_hash_table, n_compounds, w, t = params
    cmd = f'\"cd {base_path}; source ~/.zshrc && ' \
        f'python3 {base_path}/run_multi_probe_lsh_evaluation.py ' \
        f'--n-hash-table {n_hash_table} ' \
        f'--n-compounds {n_compounds} ' \
        f'--w {w} ' \
        f'--t {t} ' \
        f'\"'
    return work(cmd)


def main():
    timestamp = int(datetime.now().timestamp())
    logger.add(f'outputs/basic_multi_probe/logs/{timestamp}.log')
    try:
        with ThreadPoolExecutor(max_workers=100) as executor:
            executor.map(
                worker_basic_lsh,
                sorted(product(n_hash_table_list, n_compounds_list, w_list), key=lambda x: random.random())
            )
            executor.map(
                worker_multi_probe_lsh,
                sorted(product(n_hash_table_list, n_compounds_list, w_list, t_list), key=lambda x: random.random())
            )
    except Exception as e:
        logger.error(e)
    finally:
        result_df = pd.DataFrame.from_records(results)
        result_df.to_csv(f'outputs/basic_multi_probe/results/{timestamp}.csv', index=False)
        print(result_df)


if __name__ == '__main__':
    results = []
    main()
