test_directories = [
    "src/tests/integ",
    "src/tests/unit/s",
    "src/tests/unit/p",
    "src/tests/unit/sp",
    # FIXME: These tests are temporarily disabled to work with kfock
    # "src/tests/unit/d",
    # "src/tests/unit/sd",
    # "src/tests/unit/pd",
    # "src/tests/unit/spd",
]

binary = "build/eri_regent_test"

RED = "\033[1;31m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"

if __name__ == "__main__":
    import os, sys, subprocess

    for directory in test_directories:
        for test_case in os.listdir(directory):
            if test_case in ["fe"]:
                # FIXME:
                continue
            try:
                subprocess.check_call(
                    [binary, "-i", "{}/{}".format(directory, test_case)]
                )
            except:
                sys.stdout.write(RED)
                print("Failed on test case {}/{}".format(directory, test_case))
                sys.stdout.write(RESET)
                sys.exit(1)

    # FIXME
    for directory in test_directories[:1]:
        for test_case in os.listdir(directory):
            if test_case in ["fe"]:
                # FIXME
                continue
            try:
                subprocess.check_call(
                    [binary, "-i", "{}/{}".format(directory, test_case), "-a", "kfock"]
                )
            except:
                sys.stdout.write(RED)
                print("Failed on test case {}/{}".format(directory, test_case))
                sys.stdout.write(RESET)
                sys.exit(1)

    sys.stdout.write(GREEN)
    print("All tests passed!")
    sys.stdout.write(RESET)
