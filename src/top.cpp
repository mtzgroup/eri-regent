#include "kfock00000.h"
#include "topkfock.h"

int main(int argc, char **argv) {
  kfock00000_h_register();
  topkfock_h_register();
  legion_runtime_start(argc, argv, false);
}
