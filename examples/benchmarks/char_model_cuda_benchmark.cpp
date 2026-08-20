#include "core/tensor.h"
#include "modules/char_model.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>

int main() {
    // TODO: pick dims, build CharModel, random token-id input
    // TODO: time N trials of CPU forward()+backward()
    // TODO: move params to CUDA, replicate forward() (same as the charmodel test), time N trials
    // TODO: print CPU ms, CUDA ms, speedup
}
