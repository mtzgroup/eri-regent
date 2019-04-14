data_files = [
    # ("data/h2_6-311g.dat", "data/h2_6-311g_out.dat"),
    # ("data/b2_6-311g.dat", "data/b2_6-311g_out.dat"),
    # ("data/cl2_6-311g.dat", "data/cl2_6-311g_out.dat"),
    # ("data/f2_6-311g.dat", "data/hf26-311g_out.dat"),
    # ("data/li2_6-311g.dat", "data/hli26-311g_out.dat"),
    # ("data/o2_6-311g.dat", "data/h2o2-311g_out.dat"),
    ("data/h2o_6-311g.dat", "data/h2o_6-311g_out.dat"),
    # ("data/mg2_6-311g.dat", "data/h2mg2-311g_out.dat"),
] + [
    # (
    #     "data/small-water/h2o_%d_6-311g_reord.dat" % i,
    #     "data/small-water/h2o_%d_6-311g_reord_out.dat" % i,
    # )
    # for i in range(2, 5)
]

if __name__ == "__main__":
    # TODO: If possible, compile regent code once and run all tests
    import subprocess

    for (infile, outfile) in data_files:
        # TODO: Notify user of failed tests
        # TODO: Test multiple cpu's and partitions
        # TODO: Test gpu and cuda
        subprocess.call(["regent", "top.rg", "-i", infile, "-v", outfile], cwd="src/")
