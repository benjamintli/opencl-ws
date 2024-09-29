#include <CL/cl.h>
#include <sys/time.h>
#include <CL/opencl.hpp>
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

#define TOL (0.001)
const int TS = 16;

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

float get_random() {
    return rand() / (float)RAND_MAX;
}

std::string load_program(const std::string& input) {
    std::ifstream stream(input.c_str());
    if (!stream.is_open()) {
        throw std::invalid_argument("fuck");
    }

    return std::string(std::istreambuf_iterator<char>(stream),
                       (std::istreambuf_iterator<char>()));
}

template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& vec) {
    std::vector<T> flatVec;
    for (const auto& v : vec) {
        flatVec.insert(flatVec.end(), v.begin(), v.end());
    }
    return flatVec;
}

std::vector<float> createHostMatrix(int x, int y, bool initToZero) {
    std::vector<std::vector<float>> matrix(x, std::vector<float>(y, 0));
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            if (initToZero) {
                matrix[i][j] = 0;
            } else {
                matrix[i][j] = get_random();
            }
        }
    }

    return flatten(matrix);
}

int main(int argc, char** argv) {
    // host vectors
    struct timeval Tvalue;
    struct timezone dummy;
    cl::Buffer d_a;
    cl::Buffer d_b;
    cl::Buffer d_c;
    int m = std::stoi(argv[1]);
    int n = std::stoi(argv[2]);
    int k = std::stoi(argv[3]);
    std::string kernelPath = argv[4];
    std::vector<float> h_a = createHostMatrix(m, k, false);
    std::vector<float> h_b = createHostMatrix(k, n, false);
    std::vector<float> h_c = createHostMatrix(m, n, true);

    cl::Context context(DEVICE);
    cl::Program program(context, load_program(kernelPath), true);

    cl::CommandQueue queue(context);

    auto matmul =
        cl::compatibility::make_kernel<int, int, int, cl::Buffer, cl::Buffer,
                                       cl::Buffer>(program, "matmul");

    // allocate buffers
    d_a = cl::Buffer(context, begin(h_a), end(h_a), true);
    d_b = cl::Buffer(context, begin(h_b), end(h_b), true);
    d_c =
        cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * m * n);

    spdlog::info("Starting run");
    gettimeofday(&Tvalue, &dummy);
    double starttime =
        (double)Tvalue.tv_sec + 1.0e-6 * ((double)Tvalue.tv_usec);
    // run the kernel
    matmul(cl::EnqueueArgs(queue, cl::NDRange(m, n), cl::NDRange(TS, TS)), m, n, k,
           d_a, d_b, d_c);
    queue.finish();
    // finished

    gettimeofday(&Tvalue, &dummy);
    double endtime = (double)Tvalue.tv_sec + 1.0e-6 * ((double)Tvalue.tv_usec);
    double runtime = (endtime - starttime);
    double gflop = ((long)k * (long)m * (long)n * 2) / (1000 * 1000 * 1000);

    spdlog::info("Done: took {:.3f} seconds per run, {:.1f} GLOPS", runtime, gflop/runtime);
    cl::copy(queue, d_c, begin(h_c), end(h_c));

    return -1;
}
