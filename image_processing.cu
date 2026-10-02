#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <cstdlib>

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                     \
do {                                                                         \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                  << " - " << cudaGetErrorString(err) << std::endl;         \
        std::exit(EXIT_FAILURE);                                             \
    }                                                                        \
} while (0)

struct Image {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> pixels;
};

static bool readPPM(const std::string& filename, Image& img) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) return false;

    std::string magic;
    in >> magic;
    if (magic != "P6") return false;

    in >> img.width >> img.height;
    int maxval;
    in >> maxval;
    in.get();

    if (img.width <= 0 || img.height <= 0 || maxval != 255) return false;

    img.pixels.resize(static_cast<size_t>(img.width) * img.height * 3);
    in.read(reinterpret_cast<char*>(img.pixels.data()),
            static_cast<std::streamsize>(img.pixels.size()));

    return in.good() || in.gcount() == static_cast<std::streamsize>(img.pixels.size());
}

static bool writePGM(const std::string& filename, const std::vector<unsigned char>& data,
                     int width, int height) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) return false;

    out << "P5\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size()));
    return out.good();
}

__global__ void rgbToGrayKernel(const unsigned char* rgb,
                                 unsigned char* gray,
                                 int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixel = y * width + x;
    int base = pixel * 3;

    float value = 0.299f * rgb[base] +
                  0.587f * rgb[base + 1] +
                  0.114f * rgb[base + 2];

    gray[pixel] = static_cast<unsigned char>(value);
}

__global__ void sobelKernel(const unsigned char* gray,
                            unsigned char* edges,
                            int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        edges[idx] = 0;
        return;
    }

    int gx =
        -gray[(y - 1) * width + (x - 1)] +
         gray[(y - 1) * width + (x + 1)] +
        -2 * gray[y * width + (x - 1)] +
         2 * gray[y * width + (x + 1)] +
        -gray[(y + 1) * width + (x - 1)] +
         gray[(y + 1) * width + (x + 1)];

    int gy =
        -gray[(y - 1) * width + (x - 1)] -
         2 * gray[(y - 1) * width + x] -
         gray[(y - 1) * width + (x + 1)] +
         gray[(y + 1) * width + (x - 1)] +
         2 * gray[(y + 1) * width + x] +
         gray[(y + 1) * width + (x + 1)];

    int magnitude = abs(gx) + abs(gy);
    edges[idx] = static_cast<unsigned char>(magnitude > 255 ? 255 : magnitude);
}

int main(int argc, char** argv) {
    std::string inputDir = "input";
    std::string outputDir = "output";

    if (argc >= 2) inputDir = argv[1];
    if (argc >= 3) outputDir = argv[2];

    fs::create_directories(outputDir);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable GPU detected." << std::endl;
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "CUDA GPU: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Global Memory: " << prop.totalGlobalMem / (1024ULL * 1024ULL) << " MB\n";

    const dim3 block(16, 16);
    size_t imageCount = 0;
    double totalMilliseconds = 0.0;

    auto overallStart = std::chrono::high_resolution_clock::now();

    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.path().extension() != ".ppm") continue;

        Image img;
        if (!readPPM(entry.path().string(), img)) {
            std::cerr << "Skipping invalid image: " << entry.path() << "\n";
            continue;
        }

        size_t rgbBytes = img.pixels.size();
        size_t grayBytes = static_cast<size_t>(img.width) * img.height;

        unsigned char *d_rgb = nullptr, *d_gray = nullptr, *d_edges = nullptr;
        CUDA_CHECK(cudaMalloc(&d_rgb, rgbBytes));
        CUDA_CHECK(cudaMalloc(&d_gray, grayBytes));
        CUDA_CHECK(cudaMalloc(&d_edges, grayBytes));

        CUDA_CHECK(cudaMemcpy(d_rgb, img.pixels.data(), rgbBytes,
                              cudaMemcpyHostToDevice));

        dim3 grid((img.width + block.x - 1) / block.x,
                  (img.height + block.y - 1) / block.y);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));

        rgbToGrayKernel<<<grid, block>>>(d_rgb, d_gray, img.width, img.height);
        CUDA_CHECK(cudaGetLastError());

        sobelKernel<<<grid, block>>>(d_gray, d_edges, img.width, img.height);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        totalMilliseconds += milliseconds;

        std::vector<unsigned char> edges(grayBytes);
        CUDA_CHECK(cudaMemcpy(edges.data(), d_edges, grayBytes,
                              cudaMemcpyDeviceToHost));

        std::string outputName = outputDir + "/" +
            entry.path().stem().string() + "_edges.pgm";

        if (!writePGM(outputName, edges, img.width, img.height)) {
            std::cerr << "Failed to write " << outputName << "\n";
        }

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_rgb));
        CUDA_CHECK(cudaFree(d_gray));
        CUDA_CHECK(cudaFree(d_edges));

        ++imageCount;
    }

    auto overallEnd = std::chrono::high_resolution_clock::now();
    double totalSeconds =
        std::chrono::duration<double>(overallEnd - overallStart).count();

    std::cout << "\nProcessing complete.\n";
    std::cout << "Images processed: " << imageCount << "\n";
    std::cout << "CUDA kernel time: " << std::fixed << std::setprecision(3)
              << totalMilliseconds << " ms\n";
    std::cout << "Total execution time: " << std::fixed << std::setprecision(3)
              << totalSeconds << " seconds\n";

    if (imageCount > 0) {
        std::cout << "Average CUDA kernel time/image: "
                  << std::fixed << std::setprecision(3)
                  << totalMilliseconds / imageCount << " ms\n";
    }

    std::cout << "Output directory: " << outputDir << "\n";
    return EXIT_SUCCESS;
}
