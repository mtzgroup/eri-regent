test_directories = [
    "src/tests/integ",
    "src/tests/unit/s",
    "src/tests/unit/p",
    "src/tests/unit/sp",
    "src/tests/unit/d",
    "src/tests/unit/sd",
    "src/tests/unit/pd",
    "src/tests/unit/spd",
]

binary = "build/jfock_test"

RED = "\033[1;31m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"

if __name__ == "__main__":
    import os, sys, subprocess

    for directory in test_directories:
        for test_case in os.listdir(directory):
            try:
                subprocess.check_call(
                    [binary, "-i", "{}/{}".format(directory, test_case)]
                )
            except:
                sys.stdout.write(RED)
                print("Failed on test case {}/{}".format(directory, test_case))
                sys.stdout.write(RESET)
                sys.exit(1)

    sys.stdout.write(GREEN)
    print("All tests passed!")
    sys.stdout.write(RESET)
