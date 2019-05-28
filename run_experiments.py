import subprocess
from itertools import product
from concurrent.futures import ThreadPoolExecutor
import time
import pandas as pd
from loguru import logger


base_path = '/home/lizytalk/Projects/lsh/'
server_list = [f'cpu{i}' for i in range(1, 11)]
available_server_set = set(server_list)


n_hash_table_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
n_compounds_list = [1, 2, 4, 8, 16, 32, 64]
w_list = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
t_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
# n_hash_table_list = [64, ]
# n_compounds_list = [1, ]
# w_list = [0.5, 1.0, ]
# t_list = [128, ]


def worker_basic_lsh(params):
    n_hash_table, n_compounds, w = params
    while not available_server_set:
        time.sleep(1)
    server = available_server_set.pop()
    cmd = f'ssh {server} \"cd {base_path}; source ~/.zshrc && ' \
        f'python3 {base_path}/run_basic_lsh_evaluation.py ' \
        f'--n-hash-table {n_hash_table} ' \
        f'--n-compounds {n_compounds} ' \
        f'--w {w} ' \
        f'\"'
    logger.debug(f"command: {cmd}")
    output = subprocess.check_output(cmd, shell=True)
    # logger.debug(output)
    ret = eval(output)
    # ret = {}
    available_server_set.add(server)
    logger.debug(f"ret: {ret}")
    logger.debug(f"release {server}")
    return ret


def worker_multi_probe_lsh(params):
    n_hash_table, n_compounds, w, t = params
    while not available_server_set:
        time.sleep(1)
    server = available_server_set.pop()
    cmd = f'ssh {server} cd {base_path} && python3 {base_path}/run_multi_probe_lsh_evaluation.py ' \
        f'--n-hash-table {n_hash_table} ' \
        f'--n-compounds {n_compounds} ' \
        f'--w {w} ' \
        f'--t {t} '
    logger.debug(f"command: {cmd}")
    ret = eval(subprocess.check_output(cmd, shell=True))
    # ret = {}
    available_server_set.add(server)
    logger.debug(f"ret: {ret}")
    return ret


results = []
with ThreadPoolExecutor(max_workers=10) as executor:
    results_basic_lsh = executor.map(worker_basic_lsh, product(n_hash_table_list, n_compounds_list, w_list))
results.extend(results_basic_lsh)
with ThreadPoolExecutor(max_workers=10) as executor:
    results_multi_probe_lsh = executor.map(
        worker_multi_probe_lsh, product(n_hash_table_list, n_compounds_list, w_list, t_list))
results.extend(results_multi_probe_lsh)

result_df = pd.DataFrame.from_records(results)
result_df.to_csv('outputs/results.csv', index=False)
print(result_df)

