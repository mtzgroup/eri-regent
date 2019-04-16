data_files = [
    "data/h2o_6-311g.dat",
    # "data/small-water/h2o_2_6-311g.dat",
    # "data/small-water/h2o_3_6-311g.dat",
    # "data/small-water/h2o_4_6-311g.dat",
    # "data/water-boxes/h2o_5_6-311g.dat",
    # "data/water-boxes/h2o_10_6-311g.dat",
    # "data/water-boxes/h2o_50_6-311g.dat",
    # "data/water-boxes/h2o_100_6-311g.dat",
    # "data/water-boxes/h2o_250_6-311g.dat",
    # "data/water-boxes/h2o_500_6-311g.dat",
    # "data/water-boxes/h2o_750_6-311g.dat",
    # "data/water-boxes/h2o_1000_6-311g.dat",
]

def time_regent(file, num_cpus, num_gpus, num_trials):
    import subprocess, re
    regent_args = (
        "-i %s " % file
        + "--trials %d " % num_trials
    )
    legion_args = (
        "-ll:cpu %d " % num_cpus
        + "-ll:gpu %d " % num_gpus
        + "-ll:csize 1024 "
        + "-ll:fsize 1024 "
    )
    output = subprocess.check_output(
        "regent top.rg " + regent_args + legion_args, cwd="src/", shell=True
    )
    pattern = re.compile("Coulomb operator: ([0-9.]+) sec")
    return map(float, pattern.findall(output))

if __name__ == "__main__":
    import numpy as np

    # TODO: Read in options
    # TODO: Iterate over number of gpus
    # TODO: Pickle results to plot later
    num_cpus = 1
    num_gpus = 0
    num_trials = 2
    for file in data_files:
        times = time_regent(file, num_cpus, num_gpus, num_trials)
        print("File %s took %f seconds" % (file, np.mean(times)))