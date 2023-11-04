#include <cuda_runtime.h>
#include <cufft.h>
#include "aquila/global.h"
#include "aquila/source/generator/SineGenerator.h"
#include "aquila/tools/TextPlot.h"
#include <algorithm>
#include <functional>

void process_signal_with_NPP(const std::vector<double>& signal) {
    int size = signal.size();

    // Allocate device memory
    float* d_data;
    cudaMalloc(&d_data, size * sizeof(float));

    // Transfer signal to device
    cudaMemcpy(d_data, signal.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // Use NPP functions or your custom kernels for processing
    some_kernel<<<(size + 255) / 256, 256>>>(d_data, size);

    // Copy the result back (if needed)
    std::vector<float> processed_signal(size);
    cudaMemcpy(processed_signal.data(), d_data, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data);
}

int main() {
    // input signal parameters
    const std::size_t SIZE = 64;
    const Aquila::FrequencyType sampleFreq = 2000;
    const Aquila::FrequencyType f1 = 96, f2 = 813;
    const Aquila::FrequencyType f_lp = 500;

    Aquila::SineGenerator sineGenerator1(sampleFreq);
    sineGenerator1.setAmplitude(32).setFrequency(f1).generate(SIZE);
    Aquila::SineGenerator sineGenerator2(sampleFreq);
    sineGenerator2.setAmplitude(8).setFrequency(f2).setPhase(0.75).generate(SIZE);
    auto sum = sineGenerator1 + sineGenerator2;

    Aquila::TextPlot plt("Signal waveform before filtration");
    plt.plot(sum);

    // Compute FFT using CUDA
    cufftComplex* spectrum = computeFFTWithCUDA(sum.toArray(), SIZE);

    // The previous function (for simplicity) returns magnitude. You might want to 
    // change that if you need phase information.

    // create a low-pass filter in the frequency domain
    cufftComplex filterSpectrum[SIZE];
    for (std::size_t i = 0; i < SIZE; ++i) {
        if (i < (SIZE * f_lp / sampleFreq)) {
            filterSpectrum[i].x = 1.0;
            filterSpectrum[i].y = 0.0; // Imaginary part is 0
        } else {
            filterSpectrum[i].x = 0.0;
            filterSpectrum[i].y = 0.0;
        }
    }

    // Multiply the FFT of the signal with the filter spectrum
    for (std::size_t i = 0; i < SIZE; ++i) {
        cufftComplex product;
        product.x = spectrum[i].x * filterSpectrum[i].x - spectrum[i].y * filterSpectrum[i].y;
        product.y = spectrum[i].x * filterSpectrum[i].y + spectrum[i].y * filterSpectrum[i].x;
        spectrum[i] = product;
    }

    // Compute inverse FFT using CUDA to get the filtered signal
    double* filteredSignal = computeInverseFFTWithCUDA(spectrum, SIZE);

    plt.setTitle("Signal waveform after filtration");
    plt.plot(filteredSignal, SIZE);

    // Clean up resources
    delete[] spectrum;
    delete[] filteredSignal;

    return 0;
}