import os
import subprocess
import shutil
import signal
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hol_utils import (
    calc_mu_S,
    calc_alpha_sym,
    calc_unsat_p_fsolve,
    calc_sat_p_fsolve,
    calc_access_delay_s,
    calc_access_delay_u
)

def control_c(signum, frame):
    print("Exiting...")
    sys.exit(1)

signal.signal(signal.SIGINT, control_c)

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

def get_single_link_analysis(n_sld, lambda_sld, W, K, tt, tf):
    lambda_scaled = lambda_sld * tt
    state, p, thpt, qd, ad, ad_2nd = get_single_link_analysis_one_group(
        n_sld, lambda_scaled, W, K, tt, tf
    )
    
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

def main():
    dirname = 'wifi-dcf'
    ns3_path = os.path.join('../../../../ns3')

    if not os.path.exists(ns3_path):
        sys.exit(1)

    results_dir = os.path.join(os.getcwd(), 'results', f"{dirname}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(results_dir, exist_ok=True)

    os.chdir('../../../../')

    check_and_remove('wifi-dcf.dat')

    rng_run = 1
    max_packets = 1500
    W = 16
    K = 6
    tt = 31.6889
    tf = 27.2444
    sta_list = [5, 10, 15]
    min_lambda_exp = -4
    max_lambda_exp = -1
    step_size = 0.2
    lambdas = [10 ** exp for exp in np.arange(min_lambda_exp, max_lambda_exp + step_size, step_size)]

    output_dir = os.path.join(results_dir, "analysis_results")
    os.makedirs(output_dir, exist_ok=True)

    summary_file = os.path.join(output_dir, "summary.txt")
    with open(summary_file, 'w') as f_summary:
        f_summary.write("STA,lambda_sld,state_ns3,throughput_ns3_Mbps,queuing_delay_ns3_ms,access_delay_ns3_ms,e2e_delay_ns3_ms,"
                        "state_analysis,succ_prob,throughput_analysis_Mbps,queuing_delay_analysis_ms,access_delay_analysis_ms\n")
        
        print("STA,lambda_sld,state_ns3,throughput_ns3_Mbps,queuing_delay_ns3_ms,access_delay_ns3_ms,e2e_delay_ns3_ms,"
              "state_analysis,succ_prob,throughput_analysis_Mbps,queuing_delay_analysis_ms,access_delay_analysis_ms")

        for n_sld in sta_list:
            ns3_results = []
            analysis_results = []

            for lambda_sld in lambdas:
                cmd = f"./ns3 run 'single-bss-sld --rngRun={rng_run} --payloadSize={max_packets} " \
                      f"--perSldLambda={lambda_sld} --nSld={n_sld}'"
                print(f"Running command: {cmd}")
                result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if result.returncode != 0:
                    print(f"Command failed with return code {result.returncode}")
                    print(f"Error message: {result.stderr.decode()}")
                    sys.exit(1)

                if not os.path.exists('wifi-dcf.dat'):
                    print(" wifi-dcf.dat failed")
                    sys.exit(1)

                with open('wifi-dcf.dat', 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        tokens = line.strip().split(',')
                        throughput_ns3 = float(tokens[1])  
                        queuing_delay_ns3 = float(tokens[2])  
                        access_delay_ns3 = float(tokens[3])  
                        e2e_delay_ns3 = float(tokens[4])  
                        break  


                dest_filename = os.path.join(results_dir, f"wifi-dcf-lambda-{lambda_sld}-sta-{n_sld}.dat")
                move_file('wifi-dcf.dat', dest_filename)


                ns3_results.append({
                    'lambda_sld': lambda_sld,
                    'state': 'N/A', 
                    'throughput_Mbps': throughput_ns3,
                    'queuing_delay_ms': queuing_delay_ns3,
                    'access_delay_ms': access_delay_ns3,
                    'e2e_delay_ms': e2e_delay_ns3
                })


                analysis_result = get_single_link_analysis(n_sld, lambda_sld, W, K, tt, tf)
                analysis_results.append({
                    'lambda_sld': lambda_sld,
                    **analysis_result
                })


                line = f"{n_sld},{lambda_sld},{ns3_results[-1]['state']},{ns3_results[-1]['throughput_Mbps']}," \
                       f"{ns3_results[-1]['queuing_delay_ms']},{ns3_results[-1]['access_delay_ms']},{ns3_results[-1]['e2e_delay_ms']}," \
                       f"{analysis_result['state']},{analysis_result['succ_prob']},{analysis_result['throughput_Mbps']}," \
                       f"{analysis_result['queuing_delay_ms']},{analysis_result['access_delay_ms']}"
                print(line)
                f_summary.write(line + "\n")


            ns3_df = pd.DataFrame(ns3_results)
            analysis_df = pd.DataFrame(analysis_results)


            plt.figure()
            plt.title(f'Throughput Comparison (STA: {n_sld})')
            plt.xlabel('Offered Load (λ)')
            plt.ylabel('Throughput (Mbps)')
            plt.grid(True)
            plt.xscale('log')
            plt.plot(ns3_df['lambda_sld'], ns3_df['throughput_Mbps'], marker='o', linestyle='-', label='ns-3 Simulation')
            plt.plot(analysis_df['lambda_sld'], analysis_df['throughput_Mbps'], marker='s', linestyle='--', label='Analytical Model')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'throughput_comparison_sta_{n_sld}.png'))
            plt.close()


            total_delay_analysis = analysis_df['queuing_delay_ms'] + analysis_df['access_delay_ms']
            plt.figure()
            plt.title(f'E2E Delay Comparison (STA: {n_sld})')
            plt.xlabel('Offered Load (λ)')
            plt.ylabel('E2E Delay (ms)')
            plt.grid(True)
            plt.xscale('log')
            plt.plot(ns3_df['lambda_sld'], ns3_df['e2e_delay_ms'], marker='o', linestyle='-', label='ns-3 Simulation')
            plt.plot(analysis_df['lambda_sld'], total_delay_analysis, marker='s', linestyle='--', label='Analytical Model')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'e2e_delay_comparison_sta_{n_sld}.png'))
            plt.close()


            plt.figure()
            plt.title(f'Access Delay Comparison (STA: {n_sld})')
            plt.xlabel('Offered Load (λ)')
            plt.ylabel('Access Delay (ms)')
            plt.grid(True)
            plt.xscale('log')
            plt.plot(ns3_df['lambda_sld'], ns3_df['access_delay_ms'], marker='o', linestyle='-', label='ns-3 Simulation')
            plt.plot(analysis_df['lambda_sld'], analysis_df['access_delay_ms'], marker='s', linestyle='--', label='Analytical Model')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'access_delay_comparison_sta_{n_sld}.png'))
            plt.close()


            plt.figure()
            plt.title(f'Queuing Delay Comparison (STA: {n_sld})')
            plt.xlabel('Offered Load (λ)')
            plt.ylabel('Queuing Delay (ms)')
            plt.grid(True)
            plt.xscale('log')
            plt.plot(ns3_df['lambda_sld'], ns3_df['queuing_delay_ms'], marker='o', linestyle='-', label='ns-3 Simulation')
            plt.plot(analysis_df['lambda_sld'], analysis_df['queuing_delay_ms'], marker='s', linestyle='--', label='Analytical Model')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'queuing_delay_comparison_sta_{n_sld}.png'))
            plt.close()


            combined_df = pd.merge(ns3_df, analysis_df, on='lambda_sld', suffixes=('_ns3', '_analysis'))
            combined_df.to_csv(os.path.join(output_dir, f'combined_results_sta_{n_sld}.csv'), index=False)


    with open(os.path.join(results_dir, 'git-commit.txt'), 'w') as f:
        commit_info = subprocess.run(['git', 'show', '--name-only'], stdout=subprocess.PIPE)
        f.write(commit_info.stdout.decode())

    print(f"\n finished")



def check_and_remove(filename):
    if os.path.exists(filename):
        response = input(f"remove {filename}? [Yes/No]: ").strip().lower()
        if response == 'yes':
            os.remove(filename)
            print(f"removed {filename}")
        else:
            print("exist...")
            sys.exit(1)

def move_file(src_filename, dst_filename):
    if os.path.exists(dst_filename):
        os.remove(dst_filename)
    shutil.move(src_filename, dst_filename)

if __name__ == "__main__":
    main()
