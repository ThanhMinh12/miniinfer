#include <iostream>
#include <vector>
#ifdef _WIN32
#include <process.h>
#else
#include <cerrno>
#include <cstring>
#include <unistd.h>
#endif

#ifndef MINIINFER_CONVERTER_SCRIPT
#define MINIINFER_CONVERTER_SCRIPT "tools/convert_hf.py"
#endif

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: miniinfer-convert <hf-directory> <output.miniinfer> "
                 "[--quantize int8]\n";
    return 2;
  }
#ifdef _WIN32
  std::vector<const char*> arguments = {"py", "-3", MINIINFER_CONVERTER_SCRIPT};
  for (int index = 1; index < argc; ++index) arguments.push_back(argv[index]);
  arguments.push_back(nullptr);
  intptr_t status = _spawnvp(_P_WAIT, "py", arguments.data());
  return status == 0 ? 0 : 1;
#else
  std::vector<char*> arguments = {const_cast<char*>("python3"),
                                  const_cast<char*>(MINIINFER_CONVERTER_SCRIPT)};
  for (int index = 1; index < argc; ++index) arguments.push_back(argv[index]);
  arguments.push_back(nullptr);
  execvp("python3", arguments.data());
  std::cerr << "failed to start python3: " << std::strerror(errno) << "\n";
  return 1;
#endif
}
