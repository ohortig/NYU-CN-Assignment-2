#!/usr/bin/env python3
import subprocess
import re
import os
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STARTER_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

TICKS = 10000
SEED = 42

RTT_MIN_VALUES = [10, 20, 40, 50, 80, 100, 200, 500]
WINDOW_SIZES = [1, 2, 5, 10, 20]

LOSS_RATIO_NO_LOSS = 0.0
LOSS_RATIO_WITH_LOSS = 0.01


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_simulation(rtt_min: int, window_size: int, loss_ratio: float,
                   min_timeout: int = 100) -> dict:
    """Run the sliding-window simulation and return parsed results."""
    cmd = [
        "python3", "run_reliability_simulation.py",
        "--seed", str(SEED),
        "--rtt-min", str(rtt_min),
        "--ticks", str(TICKS),
        "--min-timeout", str(min_timeout),
        "--loss-ratio", str(loss_ratio),
        "sliding-window",
        "--window-size", str(window_size),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=STARTER_CODE_DIR)
    output = result.stdout + result.stderr

    match = re.search(r"Maximum in order received sequence number\s+(\d+)", output)
    if match is None:
        raise RuntimeError(
            f"Could not parse output for rtt_min={rtt_min}, W={window_size}, "
            f"loss={loss_ratio}:\n{output}"
        )
    max_seq = int(match.group(1))
    throughput = (max_seq + 1) / TICKS

    # Count retransmissions from the log lines
    retransmits = len(re.findall(r"Retransmit", output))
    transmits = len(re.findall(r"(?<!Re)Transmit", output))

    return {
        "max_seq": max_seq,
        "throughput": throughput,
        "retransmits": retransmits,
        "transmits": transmits,
        "raw_output": output,
    }


def theoretical_throughput(window_size: int, rtt_min: int, link_capacity: float = 1.0) -> float:
    """
    Predicted throughput = min(W / RTT_min, link_capacity).
    The link can carry at most 1 packet/tick, so throughput is capped there.
    """
    return min(window_size / rtt_min, link_capacity)


def divergence_pct(simulated: float, predicted: float) -> float:
    if predicted == 0:
        return 0.0
    return (predicted - simulated) / predicted * 100.0


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------
def run_experiment_grid(loss_ratio: float) -> dict:
    """Run simulations for every (window_size, rtt_min) pair.
    Returns dict keyed by (window_size, rtt_min)."""
    results = {}
    for ws in WINDOW_SIZES:
        for rtt in RTT_MIN_VALUES:
            print(f"  Running W={ws:>3d}, RTT_min={rtt:>4d}, loss={loss_ratio} ...")
            res = run_simulation(rtt, ws, loss_ratio)
            pred = theoretical_throughput(ws, rtt)
            res["predicted"] = pred
            res["divergence_pct"] = divergence_pct(res["throughput"], pred)
            results[(ws, rtt)] = res
    return results


def print_table(results: dict, label: str):
    print(f"\n{'='*90}")
    print(f"  {label}")
    print(f"{'='*90}")
    header = f"{'W':>4s} | {'RTT_min':>7s} | {'Predicted':>10s} | {'Simulated':>10s} | {'Diverg%':>8s} | {'ReTx':>6s}"
    print(header)
    print("-" * len(header))
    for ws in WINDOW_SIZES:
        for rtt in RTT_MIN_VALUES:
            r = results[(ws, rtt)]
            print(
                f"{ws:>4d} | {rtt:>7d} | {r['predicted']:>10.6f} | "
                f"{r['throughput']:>10.6f} | {r['divergence_pct']:>+7.2f}% | {r['retransmits']:>6d}"
            )
        print("-" * len(header))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_throughput_vs_inv_rtt(results: dict, loss_ratio: float, filename: str):
    """Plot throughput vs 1/RTT_min for each window size, with theoretical line."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(WINDOW_SIZES)))

    inv_rtt_fine = np.linspace(1 / max(RTT_MIN_VALUES), 1 / min(RTT_MIN_VALUES), 300)

    for idx, ws in enumerate(WINDOW_SIZES):
        inv_rtts = [1.0 / rtt for rtt in RTT_MIN_VALUES]
        tps = [results[(ws, rtt)]["throughput"] for rtt in RTT_MIN_VALUES]
        ax.plot(inv_rtts, tps, "o-", color=colors[idx], label=f"Simulated W={ws}")
        # Theoretical line (capped at link capacity = 1)
        theory = np.minimum(ws * inv_rtt_fine, 1.0)
        ax.plot(inv_rtt_fine, theory, "--", color=colors[idx], alpha=0.5)

    loss_str = f"loss={loss_ratio*100:.0f}%" if loss_ratio > 0 else "no loss"
    ax.set_xlabel(r"$1 \;/\; RTT_{min}$")
    ax.set_ylabel("Throughput (packets / tick)")
    ax.set_title(f"Sliding Window: Throughput vs $1/RTT_{{min}}$  ({loss_str})")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close(fig)


def plot_divergence_heatmap(results_loss: dict, filename: str):
    """Heatmap of throughput divergence (%) under loss, axes = W x RTT_min."""
    div_matrix = np.zeros((len(WINDOW_SIZES), len(RTT_MIN_VALUES)))
    for i, ws in enumerate(WINDOW_SIZES):
        for j, rtt in enumerate(RTT_MIN_VALUES):
            div_matrix[i, j] = results_loss[(ws, rtt)]["divergence_pct"]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(div_matrix, aspect="auto", cmap="RdYlGn_r", origin="lower")
    ax.set_xticks(range(len(RTT_MIN_VALUES)))
    ax.set_xticklabels(RTT_MIN_VALUES)
    ax.set_yticks(range(len(WINDOW_SIZES)))
    ax.set_yticklabels(WINDOW_SIZES)
    ax.set_xlabel(r"$RTT_{min}$")
    ax.set_ylabel("Window size (W)")
    ax.set_title("Divergence (%) between simulated and predicted throughput  (1% loss)")
    for i in range(len(WINDOW_SIZES)):
        for j in range(len(RTT_MIN_VALUES)):
            ax.text(j, i, f"{div_matrix[i, j]:+.1f}%", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Divergence %")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close(fig)


def plot_retransmission_analysis(filename: str):
    """
    For a case where throughput << W/RTT_min, run a short simulation and show
    the timeline of original transmissions vs retransmissions to illustrate why
    throughput drops.

    We pick a large window with small RTT_min and 1% loss to trigger many
    retransmissions.
    """
    ws_example = 20
    rtt_example = 10
    ticks_short = 2000

    cmd = [
        "python3", "run_reliability_simulation.py",
        "--seed", str(SEED),
        "--rtt-min", str(rtt_example),
        "--ticks", str(ticks_short),
        "--loss-ratio", "0.01",
        "sliding-window",
        "--window-size", str(ws_example),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=STARTER_CODE_DIR)
    output = result.stdout + result.stderr

    # Parse log events for transmit/retransmit times
    # Log format: | <tick> | <event_type> | <description> |
    tx_times = []
    retx_times = []
    for line in output.splitlines():
        match = re.match(r"\|\s*(\d+)\s*\|\s*(.*?)\s*\|", line)
        if match:
            tick = int(match.group(1))
            event_type = match.group(2).strip()
            if event_type == "Retransmit":
                retx_times.append(tick)
            elif event_type == "Transmit":
                tx_times.append(tick)

    fig, ax = plt.subplots(figsize=(12, 4))
    if tx_times:
        ax.eventplot([tx_times], lineoffsets=1, linelengths=0.5, colors="blue", label="Original Tx")
    if retx_times:
        ax.eventplot([retx_times], lineoffsets=2, linelengths=0.5, colors="red", label="Retransmission")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Event type")
    ax.set_yticks([1, 2])
    ax.set_yticklabels(["Transmit", "Retransmit"])
    ax.set_title(
        f"Tx / ReTx timeline  (W={ws_example}, RTT_min={rtt_example}, loss=1%)\n"
        "Retransmissions consume link capacity → throughput < W/RTT_min"
    )
    ax.legend(loc="upper right")
    ax.grid(True, axis="x")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis / explanation text
# ---------------------------------------------------------------------------
def print_analysis(results_no_loss: dict, results_loss: dict):
    print("\n" + "=" * 90)
    print("  ANALYSIS")
    print("=" * 90)

    # --- No-loss analysis ---
    print("\n--- No-loss scenario ---")
    print(
        "When the window size W is small enough that W/RTT_min < 1 (the link capacity),\n"
        "the throughput closely matches W/RTT_min because the sender is the bottleneck.\n"
        "When W/RTT_min >= 1, throughput is capped at 1 packet/tick (the link capacity),\n"
        "because the link becomes the bottleneck regardless of the window size."
    )

    cap_cases = []
    for ws in WINDOW_SIZES:
        for rtt in RTT_MIN_VALUES:
            pred = theoretical_throughput(ws, rtt)
            sim = results_no_loss[(ws, rtt)]["throughput"]
            if pred >= 1.0 and abs(sim - 1.0) < 0.05:
                cap_cases.append((ws, rtt))
    if cap_cases:
        examples = ", ".join(f"(W={w}, RTT={r})" for w, r in cap_cases[:5])
        print(f"  Link-capacity-capped examples: {examples}")

    below_cases = []
    for ws in WINDOW_SIZES:
        for rtt in RTT_MIN_VALUES:
            r = results_no_loss[(ws, rtt)]
            if r["divergence_pct"] > 10 and r["predicted"] < 1.0:
                below_cases.append((ws, rtt, r["divergence_pct"]))
    if below_cases:
        print("\n  Cases where throughput << W/RTT_min (divergence > 10%):")
        for ws, rtt, div in below_cases:
            print(f"    W={ws}, RTT_min={rtt}: divergence = {div:+.1f}%")
        print(
            "  This happens because the timeout mechanism triggers spurious retransmissions.\n"
            "  Each retransmission occupies the link for 1 tick, reducing the capacity available\n"
            "  for new packets. Check the retransmission timeline plot for visual evidence."
        )
    else:
        print("  No significant divergences from W/RTT_min found (protocol working correctly).")

    # --- Loss analysis ---
    print("\n--- 1% loss scenario ---")
    print(
        "Packet loss causes retransmissions, which consume link capacity and also stall\n"
        "the in-order delivery counter (since later packets cannot be counted until\n"
        "the lost packet is successfully retransmitted)."
    )

    print("\n  Divergence trends:")
    for ws in WINDOW_SIZES:
        divs = [results_loss[(ws, rtt)]["divergence_pct"] for rtt in RTT_MIN_VALUES]
        avg_div = np.mean(divs)
        print(f"    W={ws:>3d}: avg divergence = {avg_div:+.1f}%  "
              f"(range {min(divs):+.1f}% to {max(divs):+.1f}%)")

    print(
        "\n  When W is large and RTT_min is small, W/RTT_min approaches or exceeds link capacity.\n"
        "  In this regime, even a small amount of loss causes noticeable throughput degradation\n"
        "  because the pipeline is already full and retransmissions compete with new packets.\n"
        "  When W is small or RTT_min is large, the pipeline is under-utilized, so occasional\n"
        "  retransmissions have less impact on throughput."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 90)
    print("  Sliding-Window Throughput Analysis")
    print("=" * 90)

    # --- Run experiments ---
    print("\n[1/2] Running simulations WITHOUT loss ...")
    results_no_loss = run_experiment_grid(LOSS_RATIO_NO_LOSS)
    print_table(results_no_loss, "No Loss Results")

    print("\n[2/2] Running simulations WITH 1% loss ...")
    results_loss = run_experiment_grid(LOSS_RATIO_WITH_LOSS)
    print_table(results_loss, "1% Loss Results")

    # --- Plots ---
    print("\nGenerating plots ...")
    plot_throughput_vs_inv_rtt(results_no_loss, LOSS_RATIO_NO_LOSS,
                              "sliding_window_throughput_no_loss.png")
    plot_throughput_vs_inv_rtt(results_loss, LOSS_RATIO_WITH_LOSS,
                              "sliding_window_throughput_1pct_loss.png")
    plot_divergence_heatmap(results_loss, "sliding_window_divergence_heatmap.png")
    plot_retransmission_analysis("sliding_window_retransmission_timeline.png")

    # --- Analysis ---
    print_analysis(results_no_loss, results_loss)

    print("\nDone. All plots saved to:", OUTPUT_DIR)
