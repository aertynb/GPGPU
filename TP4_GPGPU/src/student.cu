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
					|| std::abs(a[i].z - b[i].z) > 2 || std::abs(a[i].w - b[i].w) > 2)
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

	__global__
	void equalizeHisto(	const uchar3 *const input, const int *const histogram, const uint imgWidth, const uint imgHeight,
						uchar3 *const output) 
	{
		const int grayScale = 256;
		const int n = imgHeight * imgWidth;
		for(int y = blockIdx.y * blockDim.y + threadIdx.y; y < imgHeight; y += gridDim.y * blockDim.y) 
		{
			for(int x = blockIdx.x * blockDim.x + threadIdx.x; x < imgWidth; x += gridDim.x * blockDim.x) 
			{
				const int idOut = y * imgWidth + x;
				const auto V = input[idOut].z;
				output[idOut] = input[idOut];
				output[idOut].z = static_cast<uchar>(
					(((grayScale - 1) * static_cast<float>(histogram[V])) / static_cast<float>(grayScale * n)) * 255
				);
			}
		}
	}

    void doNaive(	const std::vector<uchar3> &input, 
					const uint imgWidth, 
					const uint imgHeight,
					const std::vector<int> &histogram,
					std::vector<uchar3> &output)
	{
		std::cout << "====================================================== Naive" << std::endl;
		// 3 arrays for GPU
		uchar3 *dev_input = NULL;
		uchar3 *dev_output = NULL;
		int* dev_histo = NULL;

		// Allocate arrays on device (input and ouput)
		const size_t bytesImg = input.size() * sizeof(uchar3);
		const size_t bytesHisto = histogram.size() * sizeof(int);
		std::cout 	<< "Allocating input, output and convolution matrix on GPU" << bytesImg<<std::endl;
		HANDLE_ERROR(cudaMalloc((void**)&dev_input, bytesImg));
		HANDLE_ERROR(cudaMalloc((void**)&dev_output, bytesImg));
		HANDLE_ERROR(cudaMalloc((void**)&dev_histo, bytesHisto));

		// Copy data from host to device (input arrays) 
		std::cout << "Copy data from host to device" << std::endl;
		HANDLE_ERROR(cudaMemcpy(dev_input, input.data(), bytesImg, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(dev_histo, histogram.data(), bytesHisto, cudaMemcpyHostToDevice));

		// Configure kernel
		const dim3 nbThreads(32, 32);
		const dim3 nbBlocks((imgWidth + nbThreads.x - 1) / nbThreads.x, (imgHeight + nbThreads.y - 1) / nbThreads.y);

		std::cout << "Process on GPU (" << nbBlocks.x << "x" << nbBlocks.y << " blocks - " 
										<< nbThreads.x << "x" << nbThreads.y << " threads)" << std::endl;
								
		ChronoGPU chrGPU;
		chrGPU.start();
		equalizeHisto<<< nbBlocks, nbThreads >>>(dev_input, dev_histo, imgWidth, imgHeight, dev_output);
		chrGPU.stop();
		std::cout << "Done : " << (chrGPU.elapsedTime()) << "ms to complete" << std::endl;

		// Copy data from device to host (output array)   
		HANDLE_ERROR(cudaMemcpy(output.data(), dev_output, bytesImg, cudaMemcpyDeviceToHost)); 

		// for (const auto& pixel : output) {
		// 	std::cout << static_cast<int>(pixel.z) << std::endl;
		// }

		// Free arrays on device
		cudaFree(dev_input);
		cudaFree(dev_output);
		cudaFree(dev_histo);
		std::cout << "============================================================" << std::endl << std::endl;
	}
	
    void studentJob(const std::vector<uchar3> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
					const std::vector<int> &histogram,
					// const std::vector<uchar3> &resultCPU, // Just for comparison
                    std::vector<uchar3> &output // Output image
					)
	{
		doNaive(inputImg,imgWidth,imgHeight, histogram, output);
	}
}
