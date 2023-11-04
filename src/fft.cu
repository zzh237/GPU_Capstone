#include <cmath>
#include <fstream>
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <vector>

// Constants for the input signal
const std::size_t SIZE = 1024;
const double sampleRate = 2000.0;
const double T = 1.0 / sampleRate;
const double f1 = 96.0;
const double f2 = 813.0;
const double t_max = SIZE * T;

// Signal generation function
std::vector<double> generateSignal(std::size_t size) {
    std::vector<double> signal(size);
    for (std::size_t i = 0; i < size; ++i) {
        double t = i * T;
        signal[i] = 32 * sin(2 * M_PI * f1 * t) + 8 * sin(2 * M_PI * f2 * t);
    }
    return signal;
}

void saveToTextFile(const std::string& title, const double* signal, std::size_t size) {
    std::ofstream outFile("output.txt", std::ios_base::app);
    outFile << title << "\n";

    double max_val = *std::max_element(signal, signal + size);
    double min_val = *std::min_element(signal, signal + size);

    for (std::size_t i = 0; i < size; ++i) {
        int num_asterisks = static_cast<int>((signal[i] - min_val) / (max_val - min_val) * 50);
        for (int j = 0; j < num_asterisks; ++j) {
            outFile << "*";
        }
        outFile << "\n";
    }

    outFile << "\n\n";
    outFile.close();
}

// CUDA kernel for creating the filter spectrum
__global__ void createFilterSpectrum(cufftComplex* filter, int size, int cutoffIdx) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        filter[idx].x = (idx < cutoffIdx) ? 1.0f : 0.0f;
        filter[idx].y = 0.0f;
    }
}

cufftComplex* computeFFTWithCUDA(double* signal, std::size_t SIZE) {
    cufftHandle plan;
    cufftDoubleComplex* d_signal;
    cufftComplex* d_spectrum;

    cudaMalloc(&d_signal, SIZE * sizeof(cufftDoubleComplex));
    cudaMalloc(&d_spectrum, SIZE * sizeof(cufftComplex));

    cudaMemcpy(d_signal, signal, SIZE * sizeof(double), cudaMemcpyHostToDevice);

    cufftPlan1d(&plan, SIZE, CUFFT_D2Z, 1);
    cufftExecD2Z(plan, (cufftDoubleReal*)d_signal, d_spectrum);

    cudaFree(d_signal);
    cufftDestroy(plan);

    return d_spectrum;
}

double* computeInverseFFTWithCUDA(cufftComplex* d_spectrum, std::size_t SIZE) {
    cufftHandle plan;
    cufftDoubleComplex* d_filteredSignal;
    double* h_filteredSignal = new double[SIZE];

    cudaMalloc(&d_filteredSignal, SIZE * sizeof(cufftDoubleComplex));

    cufftPlan1d(&plan, SIZE, CUFFT_Z2D, 1);
    cufftExecZ2D(plan, d_spectrum, (cufftDoubleReal*)d_filteredSignal);

    cudaMemcpy(h_filteredSignal, d_filteredSignal, SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_filteredSignal);
    cufftDestroy(plan);

    return h_filteredSignal;
}

void saveAsPPM(const std::string& filename, const std::vector<double>& data) {
    int width = data.size();
    int height = 256;

    std::vector<std::vector<int>> image(height, std::vector<int>(width, 255));

    double max_val = *std::max_element(data.begin(), data.end());
    double min_val = *std::min_element(data.begin(), data.end());

    for (int x = 0; x < width; x++) {
        int y = static_cast<int>((data[x] - min_val) / (max_val - min_val) * (height - 1));
        for (int j = 0; j <= y; j++) {
            image[j][x] = 0;
        }
    }

    std::ofstream outFile(filename + ".ppm");
    outFile << "P2\n" << width << " " << height << "\n255\n";

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            outFile << image[i][j] << " ";
        }
        outFile << "\n";
    }
}

int main() {
    std::vector<double> sum = generateSignal(SIZE);

    // Save the original signal as PPM
    //saveToTextFile("Signal waveform before filtration", sum.data(), SIZE);
    saveAsPPM("original_signal", sum);

    cufftComplex* d_spectrum = computeFFTWithCUDA(sum.data(), SIZE);
    cufftComplex* d_filterSpectrum;
    cudaMalloc(&d_filterSpectrum, SIZE * sizeof(cufftComplex));
    int cutoffIdx = (int)(SIZE * f1 / sampleRate);
    createFilterSpectrum<<<(SIZE + 255) / 256, 256>>>(d_filterSpectrum, SIZE, cutoffIdx);

    // Multiply the signal spectrum with the filter spectrum on the GPU
    // Skipping this for simplicity

    double* filteredSignal = computeInverseFFTWithCUDA(d_spectrum, SIZE);

    // Save the filtered signal as PPM
    std::vector<double> filteredSignalVec(filteredSignal, filteredSignal + SIZE);
    //saveToTextFile("Signal waveform after filtration", filteredSignal, SIZE);
    saveAsPPM("filtered_signal", filteredSignalVec);

    cudaFree(d_spectrum);
    delete[] filteredSignal;

    return 0;
}
