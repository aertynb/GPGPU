/*
* TP 2 - Convolution d'images
* --------------------------
* Mémoire constante et textures
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

#define USE_CONSTANT
#define GRAYSCALE 256

namespace IMAC
{
	
// ================================================== For image comparison
	std::ostream &operator <<(std::ostream &os, const uchar4 &c)
	{
		os << "[" << uint(c.x) << "," << uint(c.y) << "," << uint(c.z) << "," << uint(c.w) << "]";  
    	return os; 
	}

	void compareImages(const std::vector<uchar4> &a, const std::vector<uchar4> &b)
	{
		bool error = false;
		if (a.size() != b.size())
		{
			std::cout << "Size is different !" << std::endl;
			error = true;
		}
		else
		{
			for (uint i = 0; i < a.size(); ++i)
			{
				// Floating precision can cause small difference between host and device
				if (	std::abs(a[i].x - b[i].x) > 2 || std::abs(a[i].y - b[i].y) > 2 
					|| std::abs(a[i].z - b[i].z) > 2)
				{
					std::cout << "Error at index " << i << ": a = " << a[i] << " - b = " << b[i] << " - " << std::abs(a[i].x - b[i].x) << std::endl;
					error = true;
					break;
				}
			}
		}
		if (error)
		{
			std::cout << " -> You failed, retry!" << std::endl;
		}
		else
		{
			std::cout << " -> Well done!" << std::endl;
		}
	}
	
	// ==================================================

	__device__ void rgbToHsv(const uchar4 rgb, uchar4 &hsv) {
		float r = rgb.x / 255.0f;
		float g = rgb.y / 255.0f;
		float b = rgb.z / 255.0f;

		float maxVal = fmaxf(r, fmaxf(g, b));
		float minVal = fminf(r, fminf(g, b));
		float delta = maxVal - minVal;

		float h = 0.0f;
		float s = (maxVal == 0.0f) ? 0.0f : (delta / maxVal);
		float v = maxVal;

		if (delta > 0.0f) {
			if (maxVal == r) {
				h = 60.0f * fmodf((g - b) / delta, 6.0f);
			} else if (maxVal == g) {
				h = 60.0f * ((b - r) / delta + 2.0f);
			} else if (maxVal == b) {
				h = 60.0f * ((r - g) / delta + 4.0f);
			}
		}

		if (h < 0.0f) {
			h += 360.0f;
		}

		hsv.x = static_cast<unsigned char>((h / 360.0f) * 255.0f);
		hsv.y = static_cast<unsigned char>(s * 255.0f);
		hsv.z = static_cast<unsigned char>(v * 255.0f);
		hsv.w = rgb.w;
	}

	

	__device__ void hsvToRgb(const uchar4 hsv, uchar4 &rgb) {
		float h = (hsv.x / 255.0f) * 360.0f; // H -> [0, 360]
		float s = hsv.y / 255.0f;            // S -> [0, 1]
		float v = hsv.z / 255.0f;            // V -> [0, 1]

		float c = v * s;
		float x = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
		float m = v - c;

		float rPrime, gPrime, bPrime;

		if (h < 60.0f) {
			rPrime = c;
			gPrime = x;
			bPrime = 0.0f;
		} else if (h < 120.0f) {
			rPrime = x;
			gPrime = c;
			bPrime = 0.0f;
		} else if (h < 180.0f) {
			rPrime = 0.0f;
			gPrime = c;
			bPrime = x;
		} else if (h < 240.0f) {
			rPrime = 0.0f;
			gPrime = x;
			bPrime = c;
		} else if (h < 300.0f) {
			rPrime = x;
			gPrime = 0.0f;
			bPrime = c;
		} else {
			rPrime = c;
			gPrime = 0.0f;
			bPrime = x;
		}

		rgb.x = static_cast<unsigned char>((rPrime + m) * 255.0f);
		rgb.y = static_cast<unsigned char>((gPrime + m) * 255.0f);
		rgb.z = static_cast<unsigned char>((bPrime + m) * 255.0f);
		rgb.w = hsv.w;
	}

	__global__ void convertHsvToRgb(const uchar4 *const input, uchar4 *const output, int imgWidth, int imgHeight) {
		for(int y = blockIdx.y * blockDim.y + threadIdx.y; y < imgHeight; y += gridDim.y * blockDim.y) 
		{
			for(int x = blockIdx.x * blockDim.x + threadIdx.x; x < imgWidth; x += gridDim.x * blockDim.x) 
			{
				const int idOut = y * imgWidth + x;
				uchar4 hsv = input[idOut];
				uchar4 rgb;
				hsvToRgb(hsv, rgb);
				output[idOut] = rgb;
			}
		}
	}
	
	__global__ void convertRgbToHsv(const uchar4 *const input, uchar4 *const output, int imgWidth, int imgHeight) {
		for(int y = blockIdx.y * blockDim.y + threadIdx.y; y < imgHeight; y += gridDim.y * blockDim.y) 
		{
			for(int x = blockIdx.x * blockDim.x + threadIdx.x; x < imgWidth; x += gridDim.x * blockDim.x) 
			{
				const int idOut = y * imgWidth + x;
				uchar4 rgb = input[idOut];
				uchar4 hsv;
				rgbToHsv(rgb, hsv);
				output[idOut] = hsv;
			}
		}
	}

	__global__ 
	void initializeHisto(int *const histogram) {
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < GRAYSCALE) {
			histogram[idx] = 0;
		}
	}

	__global__ 
	void fillHisto(const uchar4 *const input, const uint imgWidth, const uint imgHeight, int *const histogram) {
		for(int y = blockIdx.y * blockDim.y + threadIdx.y; y < imgHeight; y += gridDim.y * blockDim.y) 
		{
			for(int x = blockIdx.x * blockDim.x + threadIdx.x; x < imgWidth; x += gridDim.x * blockDim.x) 
			{
				const int V = input[y * imgWidth + x].z;
				atomicAdd(&histogram[V], 1);
			}
		}
	}

	__global__
	void calculateCumulativeHistogram(int *const histogram) {
		const int idx = threadIdx.x;

		__shared__ int temp[256];
		if (idx < GRAYSCALE) {
			temp[idx] = histogram[idx];
		}
		__syncthreads();

		for (int offset = 1; offset < GRAYSCALE; offset *= 2) {
			int val = 0;
			if (idx >= offset) {
				val = temp[idx - offset];
			}
			__syncthreads();
			temp[idx] += val;
			__syncthreads();
		}

		if (idx < GRAYSCALE) {
			histogram[idx] = temp[idx];
		}
	}

	__global__
	void equalizeHisto(	const uchar4 *const input, const int *const histogram, const uint imgWidth, const uint imgHeight,
						uchar4 *const output) 
	{
		const int n = imgHeight * imgWidth;
		for(int y = blockIdx.y * blockDim.y + threadIdx.y; y < imgHeight; y += gridDim.y * blockDim.y) 
		{
			for(int x = blockIdx.x * blockDim.x + threadIdx.x; x < imgWidth; x += gridDim.x * blockDim.x) 
			{
				const int idOut = y * imgWidth + x;
				const auto V = input[idOut].z;
				output[idOut].z = static_cast<uchar>((((GRAYSCALE - 1) * static_cast<float>(histogram[V])) / static_cast<float>(GRAYSCALE * n)) * 255);
			}
		}
	}

    void doNaive(	const std::vector<uchar4> &input, 
				const uint imgWidth, 
				const uint imgHeight,
				std::vector<int> &histogram,
				const std::vector<int> &histoCPU,
				const std::vector<uchar4> &resultCPU,
				std::vector<uchar4> &output)
	{
		std::cout << "====================================================== Naive" << std::endl;
		// 3 arrays for GPU
		uchar4 *dev_input = NULL;
		uchar4 *dev_output = NULL;
		uchar4 *dev_outputHSV = NULL;
		int* dev_histo = NULL;

		const size_t bytesImg = input.size() * sizeof(uchar4);
		const size_t bytesHisto = histogram.size() * sizeof(int);
		std::cout 	<< "Allocating input, output, and histogram on GPU" << bytesImg << std::endl;
		HANDLE_ERROR(cudaMalloc((void**)&dev_input, bytesImg));
		HANDLE_ERROR(cudaMalloc((void**)&dev_output, bytesImg));
		HANDLE_ERROR(cudaMalloc((void**)&dev_outputHSV, bytesImg));
		HANDLE_ERROR(cudaMalloc((void**)&dev_histo, bytesHisto));

		std::cout << "Copy data from host to device" << std::endl;
		HANDLE_ERROR(cudaMemcpy(dev_input, input.data(), bytesImg, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(dev_histo, histogram.data(), bytesHisto, cudaMemcpyHostToDevice));

		const dim3 nbThreads(32, 32);
		const dim3 nbBlocks((imgWidth + nbThreads.x - 1) / nbThreads.x, (imgHeight + nbThreads.y - 1) / nbThreads.y);

		std::cout << "Process on GPU (" << nbBlocks.x << "x" << nbBlocks.y << " blocks - " 
										<< nbThreads.x << "x" << nbThreads.y << " threads)" << std::endl;
									
		ChronoGPU chrGPU;
		chrGPU.start();

		convertRgbToHsv<<< nbBlocks, nbThreads >>>(dev_input, dev_outputHSV, imgWidth, imgHeight);
		chrGPU.stop();
		std::cout << "Done converting to HSV: " << (chrGPU.elapsedTime()) << "ms to complete" << std::endl;

		chrGPU.start();
		
		initializeHisto<<< 1, 256 >>>(dev_histo);
		fillHisto<<< nbBlocks, nbThreads >>>(dev_outputHSV, imgWidth, imgHeight, dev_histo);
		calculateCumulativeHistogram<<< 1, 256 >>>(dev_histo);

		chrGPU.stop();
		std::cout << "Done calculating cumulative histogram: " << (chrGPU.elapsedTime()) << "ms to complete" << std::endl;

		chrGPU.start();

		equalizeHisto<<< nbBlocks, nbThreads >>>(dev_outputHSV, dev_histo, imgWidth, imgHeight, dev_outputHSV);

		chrGPU.stop();
		std::cout << "Done equalizing: " << (chrGPU.elapsedTime()) << "ms to complete" << std::endl;

		chrGPU.start();

		convertHsvToRgb<<< nbBlocks, nbThreads >>>(dev_outputHSV, dev_output, imgWidth, imgHeight);

		chrGPU.stop();
		std::cout << "Done converting back to RGB: " << (chrGPU.elapsedTime()) << "ms to complete" << std::endl;

		// Copy data from device to host
		HANDLE_ERROR(cudaMemcpy(output.data(), dev_outputHSV, bytesImg, cudaMemcpyDeviceToHost)); 
		HANDLE_ERROR(cudaMemcpy(histogram.data(), dev_histo, bytesHisto, cudaMemcpyDeviceToHost)); 

		compareImages(output, resultCPU);

		HANDLE_ERROR(cudaMemcpy(output.data(), dev_output, bytesImg, cudaMemcpyDeviceToHost)); 

		// for (auto i = 0; i < histoCPU.size(); ++i) {
		// 	std::cout << "CPU : " << histoCPU[i] << " | GPU : " << histogram[i] << std::endl;
		// }

		// Free arrays on device
		cudaFree(dev_input);
		cudaFree(dev_output);
		cudaFree(dev_outputHSV);
		cudaFree(dev_histo);
		std::cout << "============================================================" << std::endl << std::endl;
	}

	
    void studentJob(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
					std::vector<int> &histogram,
					const std::vector<int> &histoCPU,
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					)
	{
		doNaive(inputImg,imgWidth,imgHeight, histogram, histoCPU, resultCPU, output);
	}
}
