data_files = [
    (1, "data/h2o_6-311g.dat"),
    (2, "data/small-water/h2o_2_6-311g.dat"),
    (3, "data/small-water/h2o_3_6-311g.dat"),
    (4, "data/small-water/h2o_4_6-311g.dat"),
    (5, "data/water-boxes/h2o_5_6-311g.dat"),
    (10, "data/water-boxes/h2o_10_6-311g.dat"),
    (50, "data/water-boxes/h2o_50_6-311g.dat"),
    (100, "data/water-boxes/h2o_100_6-311g.dat"),
    # (250, "data/water-boxes/h2o_250_6-311g.dat"),
    # (500, "data/water-boxes/h2o_500_6-311g.dat"),
    # (750, "data/water-boxes/h2o_750_6-311g.dat"),
    # (1000, "data/water-boxes/h2o_1000_6-311g.dat"),
]


def time_regent(file, num_cpus, num_gpus, num_trials):
    import subprocess, re

    regent_args = "-i %s " % file + "--trials %d " % num_trials
    legion_args = (
        "-ll:cpu %d " % num_cpus
        + "-ll:gpu %d " % num_gpus
        + "-ll:csize 1024 "
        + "-ll:fsize 1024 "
        + "-fflow 0 "
    )
    output = subprocess.check_output(
        "regent top.rg " + regent_args + legion_args, cwd="src/", shell=True
    )
    pattern = re.compile("Coulomb operator: ([0-9.]+) sec")
    return map(float, pattern.findall(output))


def plot_timings(num_molecules, timings, title, file):
    from matplotlib import pyplot as plt

    plt.title(title)
    plt.xlabel("Number of Water Molecules")
    plt.ylabel("Runtime")
    plt.errorbar(
        num_molecules,
        np.mean(timings, axis=1),
        fmt="ko",
        yerr=(np.min(timings, axis=1), np.max(timings, axis=1)),
    )
    plt.savefig(file)


if __name__ == "__main__":
    import numpy as np
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark code.")
    parser.add_argument(
        "--num_cpus", default=1, type=int, help="The number of compute CPUs to use."
    )
    parser.add_argument(
        "--num_gpus", default=1, type=int, help="The number of GPUs to use."
    )
    parser.add_argument(
        "--num_trials", default=1, type=int, help="Repeat each test NUM_TRIALS times."
    )
    parser.add_argument(
        "--savefig_file", default=None, metavar="FILE", help="Save figure to FILE."
    )
    args = parser.parse_args()

    # TODO: Iterate over number of gpus
    # TODO: Pickle results to plot later
    print(
        "Running with %d CPUs and %d GPUs for %d trials"
        % (args.num_cpus, args.num_gpus, args.num_trials)
    )
    num_molecules, timings = [], []
    for n, file in data_files:
        runtimes = time_regent(file, args.num_cpus, args.num_gpus, args.num_trials)
        num_molecules.append(n)
        timings.append(runtimes)
        print("Average runtime for %d water molecules: %f" % (n, np.mean(runtimes)))

    if args.savefig_file is not None:
        plot_timings(
            num_molecules,
            timings,
            "ERI with %d CPUs and %d GPUs" % (args.num_cpus, args.num_gpus),
            args.savefig_file,
        )
