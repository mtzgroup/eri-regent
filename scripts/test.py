test_files = [
    ("data/h2", "data/h2/output.dat", "S"),
    ("data/h2o", "data/h2o/output.dat", "P"),
    ("data/co2", "data/co2/output.dat", "P"),
    ("data/fe", "data/fe/output.dat", "D"),
]

quick_test_files = test_files[:1]

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

    for (indirectory, outfile, L) in data_files:
        # TODO: Test multiple cpu's and partitions
        # TODO: Test gpu and cuda
        try:
            subprocess.check_call(
                [
                    "regent",
                    "top.rg",
                    "-L",
                    str(L),
                    "-i",
                    indirectory,
                    "-v",
                    outfile,
                    "-fflow",
                    "0",
                ],
                cwd="src/",
            )
        except:
            sys.stdout.write(RED)
            print("Failed on input " + indirectory)
            sys.stdout.write(RESET)
            sys.exit(1)

    sys.stdout.write(GREEN)
    print("All tests passed!")
    sys.stdout.write(RESET)
