molecule_name = "H2O"
data_files = [
    # (1, "data/h2o_6-311g.dat"),
    # (2, "data/small-water/h2o_2_6-311g.dat"),
    # (3, "data/small-water/h2o_3_6-311g.dat"),
    # (4, "data/small-water/h2o_4_6-311g.dat"),
    # (5, "data/water-boxes/h2o_5_6-311g.dat"),
    # (10, "data/water-boxes/h2o_10_6-311g.dat"),
    (50, "data/water-boxes/h2o_50_6-311g.dat"),
    (100, "data/water-boxes/h2o_100_6-311g.dat"),
    (250, "data/water-boxes/h2o_250_6-311g.dat"),
    # (500, "data/water-boxes/h2o_500_6-311g.dat"),
    # (750, "data/water-boxes/h2o_750_6-311g.dat"),
    # (1000, "data/water-boxes/h2o_1000_6-311g.dat"),
]
# molecule_name = "H"
# data_files = [
#     (2 ** 3, "data/H-grids/2x2x2_6-311g.dat"),
#     (3 ** 3, "data/H-grids/3x3x3_6-311g.dat"),
#     (4 ** 3, "data/H-grids/4x4x4_6-311g.dat"),
#     (5 ** 3, "data/H-grids/5x5x5_6-311g.dat"),
#     (6 ** 3, "data/H-grids/6x6x6_6-311g.dat"),
#     (7 ** 3, "data/H-grids/7x7x7_6-311g.dat"),
#     (8 ** 3, "data/H-grids/8x8x8_6-311g.dat"),
# ]


def time_regent(file, num_gpus, num_trials):
    import subprocess, re

    regent_args = "-i %s " % file + "-p %d " % num_gpus + "--trials %d " % num_trials
    legion_args = (
        "-ll:cpu 1 -ll:util 1 -ll:gpu %d -ll:csize 1024 -ll:fsize 1024 " % num_gpus
    )
    output = subprocess.check_output(
        "regent top.rg " + regent_args + legion_args, cwd="src/", shell=True
    )
    pattern = re.compile("Coulomb operator: ([0-9.]+) sec")
    return map(float, pattern.findall(output))


def plot_timings(timings_data):
    from matplotlib import pyplot as plt

    for num_molecules, experiments in sorted(timing_data.items()):
        num_gpus = np.array([n for n, _ in experiments])
        runtimes = np.mean([r for _, r in experiments], axis=1)
        efficiency = runtimes[0] / (num_gpus * runtimes)
        plt.plot(num_gpus, efficiency, label=str(num_molecules) + " " + molecule_name)

    plt.title("Efficiency of eri-regent on GPUs")
    plt.ylabel("Efficiency")
    plt.xlabel("Number of GPUs")
    plt.xticks(num_gpus)
    plt.legend()
    plt.show()

    num_molecules = [n for n, _ in timing_data.items()]
    order = np.argsort(num_molecules)
    num_molecules = np.sort(num_molecules)
    num_gpu_trials = np.min([len(r) for _, r in timing_data.items()])
    for i in range(num_gpu_trials):
        runtimes = np.mean([r[i][1] for _, r in timing_data.items()], axis=1)
        runtimes = runtimes[order]
        num_gpus = [r[i][0] for _, r in timing_data.items()]
        for n in num_gpus:
            assert num_gpus[0] == n
        plt.plot(num_molecules, runtimes, label=str(num_gpus[0]) + " GPUs")
    plt.plot([50, 10, 250], [0.63, 2.98, 7.6], label="TeraChem")

    plt.title("Runtime of eri-regent for " + molecule_name)
    plt.ylabel("Runtime (seconds)")
    plt.xlabel("Number of " + molecule_name)
    plt.xticks(num_molecules)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    import numpy as np
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark code.")
    parser.add_argument(
        "--max_gpus", default=1, type=int, help="The maximum number of GPUs to use."
    )
    parser.add_argument(
        "--num_trials", default=1, type=int, help="Repeat each test NUM_TRIALS times."
    )
    parser.add_argument(
        "--output_file", default=None, metavar="FILE", help="Pickle results to FILE."
    )
    parser.add_argument(
        "--input_file", default=None, metavar="FILE", help="Plot pickled data in FILE."
    )
    args = parser.parse_args()

    if args.input_file is not None:
        import pickle

        with open(args.input_file, "rb") as f:
            timing_data = pickle.load(f)
            plot_timings(timing_data)
        exit(0)

    print("Running on up to %d GPUs for %d trials" % (args.max_gpus, args.num_trials))
    timing_data = dict(
        [
            (
                num_molecules,
                [
                    (num_gpus, time_regent(file, num_gpus, args.num_trials))
                    for num_gpus in range(1, args.max_gpus + 1)
                ],
            )
            for num_molecules, file in data_files
        ]
    )

    for num_molecules, experiments in timing_data.items():
        for num_gpus, runtimes in experiments:
            print(
                "Average runtime for %d %s molecules on %d GPUs: %f"
                % (num_molecules, molecule_name, num_gpus, np.mean(runtimes))
            )

    if args.output_file is not None:
        import pickle

        with open(args.output_file, "wb") as f:
            pickle.dump(timing_data, f)
