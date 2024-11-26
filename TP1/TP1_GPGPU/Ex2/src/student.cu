/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 2: Addition de vecteurs
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC
{
	__global__ void sumArraysCUDA(const int n, const int *const dev_a, const int *const dev_b, int *const dev_res)
	{
		int idx = threadIdx.x + (blockIdx.x * n); // Calcul de l'indice global
		dev_res[idx] = dev_a[idx] + dev_b[idx]; // Utiliser l'indice global
	}

    void studentJob(const int size, const int *const h_a, const int *const h_b, int *const h_res)
	{
		ChronoGPU chrGPU;

		// 3 arrays for GPU
		int *dev_a = NULL;
		int *dev_b = NULL;
		int *dev_res = NULL;
		int block_number = (size / 1024)+1; // pb ici

		// Allocate arrays on device (input and ouput)
		const size_t bytes = size * sizeof(int);
		std::cout 	<< "Allocating input (3 arrays): " 
					<< ( ( 3 * bytes ) >> 20 ) << " MB on Device" << std::endl;
		chrGPU.start();
		
		cudaMalloc((void **) &dev_a, bytes);
		cudaMalloc((void **) &dev_b, bytes);
		cudaMalloc((void **) &dev_res, bytes);
		
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from host to device (input arrays) 
		cudaMemcpy(dev_a, h_a, bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b, h_b, bytes, cudaMemcpyHostToDevice);

		//std::cout << size << std::endl;
		// Launch kernel
		sumArraysCUDA<<<block_number, size / block_number>>>(size / block_number, dev_a, dev_b, dev_res);

		// Question 5 refaire des appels kernel

		// Copy data from device to host (output array)  
		cudaMemcpy(h_res, dev_res, bytes, cudaMemcpyDeviceToHost);

		// Free arrays on device
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFree(dev_res);
	}
}
