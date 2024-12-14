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
import sys
import matplotlib.pyplot as plt
import os

# Define single link single group analysis function
def get_single_link_analysis_one_group(n, lamb, W, K, tt, tf):
    """
    Analyze the performance of a single group of devices on a single link.

    Parameters:
        n (int): Number of single link devices.
        lamb (float): Arrival rate for single link devices (scaled by tau_T).
        W (int): Initial contention window size.
        K (int): Maximum number of retransmissions.
        tt (float): Holding time tau_T.
        tf (float): Holding time tau_F.

    Returns:
        tuple: Contains state, success probability, throughput, queuing delay,
               access delay, and second raw moment of access delay.
    """
    p_unsat, flag_unsat = calc_unsat_p_fsolve(n, lamb, tt, tf)
    if flag_unsat:
        alpha_u = calc_alpha_sym(tt, tf, n, p_unsat)
        queueing_delay, access_delay, access_delay_2nd_raw_moment = calc_access_delay_u(
            p_unsat, alpha_u, tt, tf, W, K, lamb
        )
        queueing_delay = np.inf if queueing_delay < 0 or queueing_delay > 1e9 else queueing_delay
        state = "U"  # Unsaturated
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
        state = "S"  # Saturated
        thpt = min(lamb, pi_ts) * n
        return state, p_sat, thpt, queueing_delay, access_delay, access_delay_2nd_raw_moment

# Define single link analysis function
def get_single_link_analysis(n_sld, lambda_sld, W, K, tt, tf):
    """
    Analyze the performance metrics of a single link with a single group of devices.

    Parameters:
        n_sld (int): Number of single link devices.
        lambda_sld (float): Arrival rate for single link devices (per node per slot).
        W (int): Initial contention window size.
        K (int): Maximum number of retransmissions.
        tt (float): Holding time tau_T.
        tf (float): Holding time tau_F.

    Returns:
        dict: Contains state, success probability, throughput, queuing delay,
              access delay, and second moments of access delay.
    """
    lambda_scaled = lambda_sld * tt  # Scale arrival rate
    state, p, thpt, qd, ad, ad_2nd = get_single_link_analysis_one_group(
        n_sld, lambda_scaled, W, K, tt, tf
    )
    
    # Unit conversion
    thpt_mbps = thpt / tt * 1500 * 8 / 9  # Convert to Mbps
    queuing_delay_ms = qd * 0.009  # Convert to ms
    access_delay_ms = ad * 0.009  # Convert to ms
    ad_2nd_raw_moment = ad_2nd * (0.009 ** 2)  # Convert to ms^2
    ad_2nd_central_moment = (ad_2nd - (ad ** 2)) * (0.009 ** 2)  # Convert to ms^2
    
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
    # Experiment parameters
    n_sld = 15  # Number of single link devices, consistent with ns-3 simulation
    W = 16
    K = 6
    tt = 31.6889
    tf = 27.2444

    # Arrival rate range, similar to ns-3 simulation lambdas
    min_lambda_exp = -4
    max_lambda_exp = -1
    step_size = 0.2
    lambdas = [10 ** exp for exp in np.arange(min_lambda_exp, max_lambda_exp + step_size, step_size)]

    # Store results
    results = []

    # Prepare output directory
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)

    # Open summary text file
    summary_file = os.path.join(output_dir, "summary.txt")
    with open(summary_file, 'w') as f_summary:
        # Write header
        f_summary.write("lambda_sld, state, succ_prob, throughput_Mbps, queuing_delay_ms, access_delay_ms, "
                        "access_delay_2nd_raw_moment_ms2, access_delay_2nd_central_moment_ms2\n")
        
        # Print header to console
        print("lambda_sld, state, succ_prob, throughput_Mbps, queuing_delay_ms, access_delay_ms, "
              "access_delay_2nd_raw_moment_ms2, access_delay_2nd_central_moment_ms2")

        for lambda_sld in lambdas:
            result = get_single_link_analysis(n_sld, lambda_sld, W, K, tt, tf)
            results.append(result)
            line = f"{lambda_sld},{result['state']},{result['succ_prob']},{result['throughput_Mbps']},{result['queuing_delay_ms']},{result['access_delay_ms']},{result['access_delay_2nd_raw_moment_ms2']},{result['access_delay_2nd_central_moment_ms2']}"
            print(line)
            f_summary.write(line + "\n")
    
    # Convert results to DataFrame for plotting
    df = pd.DataFrame(results)
    df['lambda_sld'] = lambdas

    # Save results to CSV
    csv_file = os.path.join(output_dir, "hol_ana_results_single_link.csv")
    df.to_csv(csv_file, index=False)

    # Plot Throughput vs. Offered Load (lambda)
    plt.figure()
    plt.title('Throughput vs. Offered Load')
    plt.xlabel('Offered Load (λ)')
    plt.ylabel('Throughput (Mbps)')
    plt.grid(True)
    plt.xscale('log')
    plt.plot(df['lambda_sld'], df['throughput_Mbps'], marker='o', linestyle='-')
    throughput_plot = os.path.join(output_dir, 'throughput_vs_lambda.png')
    plt.savefig(throughput_plot)
    plt.close()

    # Plot Queuing Delay vs. Offered Load (lambda)
    plt.figure()
    plt.title('Queuing Delay vs. Offered Load')
    plt.xlabel('Offered Load (λ)')
    plt.ylabel('Queuing Delay (ms)')
    plt.grid(True)
    plt.xscale('log')
    plt.plot(df['lambda_sld'], df['queuing_delay_ms'], marker='o', linestyle='-')
    queuing_delay_plot = os.path.join(output_dir, 'queuing_delay_vs_lambda.png')
    plt.savefig(queuing_delay_plot)
    plt.close()

    # Plot Access Delay vs. Offered Load (lambda)
    plt.figure()
    plt.title('Access Delay vs. Offered Load')
    plt.xlabel('Offered Load (λ)')
    plt.ylabel('Access Delay (ms)')
    plt.grid(True)
    plt.xscale('log')
    plt.plot(df['lambda_sld'], df['access_delay_ms'], marker='o', linestyle='-')
    access_delay_plot = os.path.join(output_dir, 'access_delay_vs_lambda.png')
    plt.savefig(access_delay_plot)
    plt.close()

    print(f"\nAnalysis complete. Results saved in the '{output_dir}' directory.")

if __name__ == "__main__":
    main()
