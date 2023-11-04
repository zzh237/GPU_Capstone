# CUDA Frequency-Domain Filtering Project

This project demonstrates the implementation of frequency-domain filtering using the Fast Fourier Transform (FFT) with CUDA. It's an adaptation of the Aquila FFT example, with updates to leverage the parallel processing capabilities of NVIDIA GPUs.

## Project Description

The frequency-domain filtering method operates on the principle that multiplication in the frequency domain corresponds to convolution in the time domain. The program inputs a signal composed of two sine waves and uses FFT to shift to the frequency domain. A low-pass filter is then applied to the signal's spectrum, and the inverse FFT is used to move back to the time domain, effectively filtering out one of the sine waves.

This CUDA-based implementation optimizes the FFT operations by leveraging the parallel processing capabilities of NVIDIA GPUs.

## Usage

1. Compile the program using the provided `Makefile`.
2. Run the executable:
