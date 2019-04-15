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

if __name__ == "__main__":
    import subprocess, re

    # TODO: Read in options
    # TODO: Iterate over number of gpus
    for file in data_files:
        legion_args = (
            "-ll:cpu 1 "
            + "-ll:gpu 0 "
            + "-ll:csize 1024 "
            + "-ll:fsize 1024 "
        )
        output = subprocess.check_output(
            "regent top.rg -i " + file + " " + legion_args, cwd="src/", shell=True
        )
        result = re.search("Coulomb operator: [0-9.]+ sec", output).group()
        time = float(re.search("[0-9.]+", result).group())

        print("File %s took %f seconds" % (file, time))
