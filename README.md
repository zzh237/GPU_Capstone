# CUDA Frequency-Domain Filtering Project

This project demonstrates the implementation of frequency-domain filtering using the Fast Fourier Transform (FFT) with CUDA. It is an adaptation of the Aquila FFT example, enhanced to take advantage of the parallel processing capabilities of NVIDIA GPUs.

## Project Description

The frequency-domain filtering method operates based on the principle that multiplication in the frequency domain corresponds to convolution in the time domain. The program inputs a signal composed of two sine waves and employs FFT to transition to the frequency domain. A low-pass filter is subsequently applied to the signal's spectrum. The inverse FFT then shifts the filtered spectrum back to the time domain, effectively isolating one of the sine waves.

This CUDA-based solution optimizes the FFT computations by leveraging the immense parallel processing potential of NVIDIA GPUs.

## Usage

1. Compile the program using the provided Makefile:

2. Run the executable:

## Example

For a demonstration, run the provided `run.sh` script, which employs the FFT filter on a sample signal.

Input and output examples can be found in `data/original_signal.ppm` and `data/filtered_signal.ppm`, respectively.

<p align="center">
  <img src="https://github.com/zzh237/GPU_Capstone/blob/main/data/original_signal.png" width="45%" />
  <img src="https://github.com/zzh237/GPU_Capstone/blob/main/data/filtered_signal.png" width="45%" /> 
</p>

> **Note**: For direct visualization on GitHub, consider converting the `.ppm` files to a more universally supported format such as `.png` or `.jpg`. Alternatively, you can provide direct download links to the `.ppm` files so users can open them using appropriate software.
