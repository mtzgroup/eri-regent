test_directories = [
    "src/tests/integ",
    # "src/tests/unit/s",
    # "src/tests/unit/p",
    # "src/tests/unit/sp",
    # "src/tests/unit/d",
    # "src/tests/unit/sd",
    # "src/tests/unit/pd",
    # "src/tests/unit/spd",
]

quick_test_directories = test_directories[:1]

RED = "\033[1;31m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"

if __name__ == "__main__":
    # TODO: If possible, compile regent code once and run all tests
    import os, sys, subprocess, argparse

    parser = argparse.ArgumentParser(description="Test for correctness.")
    parser.add_argument("--quick", action="store_true", help="Run only one test case")
    args = parser.parse_args()

    directories = quick_test_directories if args.quick else test_directories

    for directory in directories:
        for test_case in os.listdir(directory):
            # TODO: Test multiple cpu's and partitions
            # TODO: Test gpu and cuda
            try:
                subprocess.check_call(
                    [
                        "regent",
                        "top.rg",
                        "-L",
                        {"h2": "S", "h2o": "P", "co2": "P", "fe": "D"}[test_case],
                        "-i",
                        "../{}/{}".format(directory, test_case),
                        "-v",
                        "../{}/{}/output.dat".format(directory, test_case),
                        "-fflow",
                        "0",
                    ],
                    cwd="src/",
                )
            except:
                sys.stdout.write(RED)
                print("Failed on test case {}/{}".format(directory, test_case))
                sys.stdout.write(RESET)
                sys.exit(1)

    sys.stdout.write(GREEN)
    print("All tests passed!")
    sys.stdout.write(RESET)
