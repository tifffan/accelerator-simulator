import numpy as np
import matplotlib.pyplot as plt
import argparse

def read_statistics(file_path):
    """Read the global statistics file and extract means and stds."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Extract means and stds
    means = []
    stds = []
    in_means_section = False
    in_stds_section = False

    for line in lines:
        line = line.strip()
        if line.startswith("Per-Step Global Mean:"):
            in_means_section = True
            continue
        elif line.startswith("Per-Step Global Std:"):
            in_means_section = False
            in_stds_section = True
            continue
        elif line.startswith("Settings Global Mean:"):
            break

        if in_means_section and line.startswith("Step"):
            means.append([float(x) for x in line.split(":")[1].split(",")])
        elif in_stds_section and line.startswith("Step"):
            stds.append([float(x) for x in line.split(":")[1].split(",")])

    global_means = np.array(means)
    global_stds = np.array(stds)

    return global_means, global_stds

def plot_statistics(global_means, global_stds):
    """Plot global means and standard deviations as functions of step."""
    steps = np.arange(global_means.shape[0])
    labels = ["x", "y", "z", "px", "py", "pz"]

    # Plot global means
    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.plot(steps, global_means[:, i], label=f"{labels[i]} Mean")
    plt.xlabel("Step")
    plt.ylabel("Global Mean")
    plt.legend()
    plt.title("Global Means vs Step")
    plt.grid(True)
    plt.savefig("global_means.png")
    plt.show()

    # Plot global means excluding z
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(labels):
        if label != "z" and label != "pz":
            plt.plot(steps, global_means[:, i], label=f"{label} Mean")
    plt.xlabel("Step")
    plt.ylabel("Global Mean (Excluding z)")
    plt.legend()
    plt.title("Global Means vs Step (Excluding z)")
    plt.grid(True)
    plt.savefig("global_means_excluding_z_pz.png")
    plt.show()
    
    # Plot global means excluding z
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(labels):
        if label != "pz":
            plt.plot(steps, global_means[:, i], label=f"{label} Mean")
    plt.xlabel("Step")
    plt.ylabel("Global Mean (Excluding z)")
    plt.legend()
    plt.title("Global Means vs Step (Excluding pz)")
    plt.grid(True)
    plt.savefig("global_means_excluding_pz.png")
    plt.show()

    # Plot global standard deviations
    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.plot(steps, global_stds[:, i], label=f"{labels[i]} Std")
    plt.xlabel("Step")
    plt.ylabel("Global Standard Deviation")
    plt.legend()
    plt.title("Global Standard Deviations vs Step")
    plt.grid(True)
    plt.savefig("global_stds.png")
    plt.show()

    # Plot global means in log-y scale
    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.semilogy(steps, global_means[:, i], label=f"{labels[i]} Mean")
    plt.xlabel("Step")
    plt.ylabel("Global Mean (Log Scale)")
    plt.legend()
    plt.title("Global Means (Log Scale) vs Step")
    plt.grid(True, which="both", linestyle="--")
    plt.savefig("global_means_logy.png")
    plt.show()

    # Plot global standard deviations in log-y scale
    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.semilogy(steps, global_stds[:, i], label=f"{labels[i]} Std")
    plt.xlabel("Step")
    plt.ylabel("Global Standard Deviation (Log Scale)")
    plt.legend()
    plt.title("Global Standard Deviations (Log Scale) vs Step")
    plt.grid(True, which="both", linestyle="--")
    plt.savefig("global_stds_logy.png")
    plt.show()

    # Plot ratio of next-step global mean to previous-step global mean
    ratios = global_means[1:] / global_means[:-1]
    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.plot(steps[1:], ratios[:, i], label=f"{labels[i]} Ratio")
    plt.xlabel("Step")
    plt.ylabel("Ratio of Next-Step to Previous-Step Mean")
    plt.legend()
    plt.title("Next-Step to Previous-Step Mean Ratio vs Step")
    plt.grid(True)
    plt.savefig("mean_ratios.png")
    plt.show()

    # Plot log of the ratio
    log_ratios = np.log(np.abs(ratios))
    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.plot(steps[1:], log_ratios[:, i], label=f"{labels[i]} Log Ratio")
    plt.xlabel("Step")
    plt.ylabel("Log of Ratio of Next-Step to Previous-Step Mean")
    plt.legend()
    plt.title("Log of Next-Step to Previous-Step Mean Ratio vs Step")
    plt.grid(True)
    plt.savefig("log_mean_ratios.png")
    plt.show()

    # # Plot log of the ratio for steps 1 to final step
    # plt.figure(figsize=(10, 6))
    # for i in range(6):
    #     plt.plot(steps[1:], log_ratios[:, i], label=f"{labels[i]} Log Ratio")
    # plt.xlabel("Step")
    # plt.ylabel("Log of Ratio of Next-Step to Previous-Step Mean")
    # plt.legend()
    # plt.title("Log of Next-Step to Previous-Step Mean Ratio (1 to Final) vs Step")
    # plt.grid(True)
    # plt.savefig("log_mean_ratios_1_to_final.png")
    # plt.show()

    # Plot log of the ratio for steps 1 to 40
    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.plot(steps[2:41], log_ratios[1:40, i], label=f"{labels[i]} Log Ratio")
    plt.xlabel("Step")
    plt.ylabel("Log of Ratio of Next-Step to Previous-Step Mean")
    plt.legend()
    plt.title("Log of Next-Step to Previous-Step Mean Ratio (2 to 40) vs Step")
    plt.grid(True)
    plt.savefig("log_mean_ratios_2_to_40.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze global statistics from a txt file.")
    parser.add_argument("file_path", type=str, help="Path to the statistics txt file.")
    args = parser.parse_args()

    # Read and process statistics
    global_means, global_stds = read_statistics(args.file_path)

    # Plot the statistics
    plot_statistics(global_means, global_stds)

if __name__ == "__main__":
    main()
