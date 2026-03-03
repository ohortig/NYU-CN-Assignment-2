#!/usr/bin/env python3
"""
Run the Stop-And-Wait protocol for several RTT_min values and plot
throughput vs 1/RTT_min.

Theoretical throughput for Stop-And-Wait: W/RTT = 1/RTT  (one packet per RTT).
"""

import subprocess
import re
import os
import matplotlib.pyplot as plt
import numpy as np

# Directory containing the simulation scripts
STARTER_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)

TICKS = 10000
SEED = 42
RTT_MIN_VALUES = [10, 20, 40, 50, 80, 100, 150, 200, 300, 500]


def run_simulation(rtt_min: int) -> float:
    """Run the stop-and-wait simulation and return throughput (packets/tick)."""
    result = subprocess.run(
        [
            "python3", "run_reliability_simulation.py",
            "--seed", str(SEED),
            "--rtt-min", str(rtt_min),
            "--ticks", str(TICKS),
            "--min-timeout", "1000", # Set a large minimum timeout to avoid timeouts affecting the throughput
            "--loss-ratio", "0.01",
            "stop-and-wait",
        ],
        capture_output=True,
        text=True,
        cwd=STARTER_CODE_DIR,
    )
    output = result.stdout + result.stderr
    match = re.search(r"Maximum in order received sequence number\s+(\d+)", output)
    if match is None:
        raise RuntimeError(f"Could not parse output for rtt_min={rtt_min}:\n{output}")
    max_seq = int(match.group(1))
    # Number of successfully delivered packets = max_seq + 1  (seq numbers start at 0)
    throughput = (max_seq + 1) / TICKS
    return throughput


if __name__ == "__main__":
    throughputs = []
    inv_rtts = []

    for rtt_min in RTT_MIN_VALUES:
        tp = run_simulation(rtt_min)
        inv_rtt = 1.0 / rtt_min
        throughputs.append(tp)
        inv_rtts.append(inv_rtt)
        print(f"RTT_min={rtt_min:>4d}  |  1/RTT_min={inv_rtt:.4f}  |  throughput={tp:.6f} packets/tick")

    # Theoretical line: throughput = W/RTT = 1/RTT_min
    inv_rtt_theory = np.linspace(min(inv_rtts), max(inv_rtts), 200)
    throughput_theory = inv_rtt_theory  # W = 1 for stop-and-wait

    plt.figure(figsize=(8, 5))
    plt.plot(inv_rtts, throughputs, "bo-", label="Simulated throughput")
    plt.plot(inv_rtt_theory, throughput_theory, "r--", label=r"Theoretical: $W/RTT = 1/RTT_{min}$")
    plt.xlabel(r"$1 / RTT_{min}$")
    plt.ylabel("Throughput (packets / tick)")
    plt.title("Stop-And-Wait: Throughput vs $1/RTT_{min}$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stop_and_wait_throughput.png")
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"\nPlot saved to {output_path}")
