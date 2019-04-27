test_files = [
    ("data/h2_6-311g.dat", "data/h2_6-311g_out.dat"),
    ("data/b2_6-311g.dat", "data/b2_6-311g_out.dat"),
    ("data/cl2_6-311g.dat", "data/cl2_6-311g_out.dat"),
    ("data/f2_6-311g.dat", "data/f2_6-311g_out.dat"),
    ("data/li2_6-311g.dat", "data/li2_6-311g_out.dat"),
    ("data/o2_6-311g.dat", "data/o2_6-311g_out.dat"),
    ("data/h2o_6-311g.dat", "data/h2o_6-311g_out.dat"),
    ("data/mg2_6-311g.dat", "data/mg2_6-311g_out.dat"),
    ("data/small-water/h2o_2_6-311g.dat", "data/small-water/h2o_2_6-311g_out.dat"),
    ("data/small-water/h2o_3_6-311g.dat", "data/small-water/h2o_3_6-311g_out.dat"),
    ("data/small-water/h2o_4_6-311g.dat", "data/small-water/h2o_4_6-311g_out.dat"),
    # ("data/ti2_crenbl.dat", "data/ti2_crenbl_out.dat"),
    # ("data/li2_cc-pvtz.dat", "data/li2_cc-pvtz_out.dat"),
]

quick_test_files = [
    ("data/small-water/h2o_4_6-311g.dat", "data/small-water/h2o_4_6-311g_out.dat")
]

RED = "\033[1;31m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"

if __name__ == "__main__":
    # TODO: If possible, compile regent code once and run all tests
    import sys, subprocess, argparse

    parser = argparse.ArgumentParser(description="Test for correctness.")
    parser.add_argument("--quick", action="store_true", help="Run only one test case")
    args = parser.parse_args()

    data_files = quick_test_files if args.quick else test_files

    for (infile, outfile) in data_files:
        # TODO: Test multiple cpu's and partitions
        # TODO: Test gpu and cuda
        try:
            subprocess.check_call(
                ["regent", "top.rg", "-i", infile, "-v", outfile, "-fflow", "0"],
                cwd="src/",
            )
        except:
            sys.stdout.write(RED)
            print("Failed on input " + infile)
            sys.stdout.write(RESET)
            sys.exit(1)

    sys.stdout.write(GREEN)
    print("All tests passed!")
    sys.stdout.write(RESET)
