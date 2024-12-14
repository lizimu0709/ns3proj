import os
import subprocess
import shutil
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from bayes_opt import BayesianOptimization

def control_c(signum, frame):
    print("Exiting...")
    sys.exit(1)

import signal
signal.signal(signal.SIGINT, control_c)

from hol_utils import (
    calc_mu_S,
    calc_alpha_sym,
    calc_unsat_p_fsolve,
    calc_sat_p_fsolve,
    calc_access_delay_s,
    calc_access_delay_u
)

def get_single_link_analysis(n_sld, lambda_sld, CWmin, K, tt, tf):

    lambda_scaled = lambda_sld * tt
    W = CWmin
    state, p, thpt, qd, ad, ad_2nd = get_single_link_analysis_one_group(
        n_sld, lambda_scaled, W, K, tt, tf
    )
    
    # 单位转换
    thpt_mbps = thpt / tt * 1500 * 8 / 9
    queuing_delay_ms = qd * 0.009
    access_delay_ms = ad * 0.009
    ad_2nd_raw_moment = ad_2nd * (0.009 ** 2)
    ad_2nd_central_moment = (ad_2nd - (ad ** 2)) * (0.009 ** 2)
    
    return {
        "state": state,
        "succ_prob": p,
        "throughput_Mbps": thpt_mbps,
        "queuing_delay_ms": queuing_delay_ms,
        "access_delay_ms": access_delay_ms,
        "access_delay_2nd_raw_moment_ms2": ad_2nd_raw_moment,
        "access_delay_2nd_central_moment_ms2": ad_2nd_central_moment
    }

def get_single_link_analysis_one_group(n, lamb, W, K, tt, tf):
    p_unsat, flag_unsat = calc_unsat_p_fsolve(n, lamb, tt, tf)
    if flag_unsat:
        alpha_u = calc_alpha_sym(tt, tf, n, p_unsat)
        queueing_delay, access_delay, access_delay_2nd_raw_moment = calc_access_delay_u(
            p_unsat, alpha_u, tt, tf, W, K, lamb
        )
        queueing_delay = np.inf if queueing_delay < 0 or queueing_delay > 1e9 else queueing_delay
        state = "U" 
        thpt = n * lamb
        return state, p_unsat, thpt, queueing_delay, access_delay, access_delay_2nd_raw_moment
    else:
        p_sat, _ = calc_sat_p_fsolve(n, W, K)
        alpha_s = calc_alpha_sym(tt, tf, n, p_sat)
        pi_ts = calc_mu_S(p_sat, tt, tf, W, K)
        queueing_delay, access_delay, access_delay_2nd_raw_moment = calc_access_delay_s(
            p_sat, alpha_s, tt, tf, W, K, lamb, pi_ts
        )
        queueing_delay = np.inf if queueing_delay < 0 or queueing_delay > 1e9 else queueing_delay
        state = "S"
        thpt = min(lamb, pi_ts) * n
        return state, p_sat, thpt, queueing_delay, access_delay, access_delay_2nd_raw_moment

def check_and_remove(filename):
    if os.path.exists(filename):
        os.remove(filename)

def move_file(src_filename, dst_filename):
    if os.path.exists(dst_filename):
        os.remove(dst_filename)
    shutil.move(src_filename, dst_filename)

def objective_function(CWmin, K, perSldLambda):
    CWmin = int(round(CWmin))
    K = int(round(K))
    perSldLambda = 10 ** perSldLambda
    CWmax = CWmin * (2 ** K)
    
    rng_run = 1
    max_packets = 1500
    n_sld = 5
    tt = 31.6889
    tf = 27.2444
    
    cmd = f"./ns3 run 'single-bss-sld --rngRun={rng_run} --payloadSize={max_packets} " \
          f"--perSldLambda={perSldLambda} --nSld={n_sld} --acBECwmin={CWmin} --acBECwmax={CWmax}'"
    print(f"Running command: {cmd}")
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        print(f"Error message: {result.stderr.decode()}")
        return None 

    if not os.path.exists('wifi-dcf.dat'):
        print("wifi-dcf.dat failed")
        return None  


    with open('wifi-dcf.dat', 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.strip().split(',')
            throughput_ns3 = float(tokens[1])
            queuing_delay_ns3 = float(tokens[2])
            access_delay_ns3 = float(tokens[3])
            e2e_delay_ns3 = float(tokens[4])
            break

    os.remove('wifi-dcf.dat')


    epsilon = 1e-6
    objective = throughput_ns3 / (e2e_delay_ns3 + epsilon)
    return objective

def main():
    ns3_path = os.path.join('../../../../ns3')
    if not os.path.exists(ns3_path):
        sys.exit(1)
    os.chdir('../../../../')

    check_and_remove('wifi-dcf.dat')

    pbounds = {
        'CWmin': (4, 1024),
        'K': (3, 10),
        'perSldLambda': (-5, -1)
    }

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
    )

    optimizer.maximize(
        init_points=5,
        n_iter=25,
    )

    print("best:")
    print(optimizer.max)

    log_file = 'bayes_opt_log.json'
    optimizer_points = pd.DataFrame(optimizer.space.params)
    optimizer_target = pd.Series(optimizer.space.target, name='target')
    optimizer_log = pd.concat([optimizer_points, optimizer_target], axis=1)
    optimizer_log.to_json(log_file, orient='records')

    plt.figure()
    plt.plot(range(len(optimizer.space.target)), optimizer.space.target)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.title('Bayesian Optimization Progress')
    plt.savefig('bayes_opt_progress.png')
    plt.close()

if __name__ == "__main__":
    main()
