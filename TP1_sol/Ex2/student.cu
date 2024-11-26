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
	// ==================================================== Kernel
	// Goal: computes the addition of two vectors (dev_a and dev_b) of size n ans stores the result in dev_res
	// We need to loop over the data if the total number of thread is smaller than the vector size
	// Given:
	// - threadIdx.x = thread id in the block
	// - blockIdx.x = block id in the grid
	// - blockDim.x = number of threads in a block
	// - gridDim.x = number of block in the grid
	// We can find:
	// -> threadIdx.x + blockIdx.x * blockDim.x = "global" id of a thread in the grid
	// -> blockDim.x * gridDim.x = total number of threads in the grid
	// So we can loop !
	// Here follow two equivalent kernels: the first use a for, the second (commented) use a while 
	__global__ void sumArraysCUDA(const int n, const int *const dev_a, const int *const dev_b, int *const dev_res)
	{
		// Loop to compute all data (if the global number of threads is smaller than array size)
		for (int 	idx = threadIdx.x + blockIdx.x * blockDim.x; // Init with the global thread id
					idx < n; // Don't exceed data size !
					idx += blockDim.x * gridDim.x) // Add total number of threads
		{
			dev_res[idx] = dev_a[idx] + dev_b[idx];
		}
	}
	// __global__ void sumArraysCUDA(const int n, const int *const dev_a, const int *const dev_b, int *const dev_res)
	// {
	// 	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	// 	while (idx < n) 
	// 	{
	// 		dev_res[idx] = dev_a[idx] + dev_b[idx];
	// 		idx += blockDim.x * gridDim.x;
	// 	}
	// }

    void studentJob(const int size, const int *const a, const int *const b, int *const res)
	{
		ChronoGPU chrGPU;

		// 3 arrays for GPU
		int *dev_a = NULL;
		int *dev_b = NULL;
		int *dev_res = NULL;

		// Allocate arrays on device (input and ouput)
		const size_t bytes = size * sizeof(int); // size of the array in bytes
		std::cout 	<< "Allocating input (3 arrays): " 
					<< ( ( 3 * bytes ) >> 20 ) << " MB on Device" << std::endl; // print used memory in MB
		// Start chrono
		chrGPU.start();
		HANDLE_ERROR( cudaMalloc( (void**)&dev_a, bytes ) );
		HANDLE_ERROR( cudaMalloc( (void**)&dev_b, bytes ) );
		HANDLE_ERROR( cudaMalloc( (void**)&dev_res, bytes ) );	
		// Stop chrono
		chrGPU.stop();
		// Print measured time
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from host to device (input arrays) 
		std::cout << "Copy inputs from host to device" << std::endl;
		chrGPU.start();
		HANDLE_ERROR( cudaMemcpy( dev_a, a, bytes, cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy( dev_b, b, bytes, cudaMemcpyHostToDevice ) );
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Configure number of threads/blocks
		// Set number of threads to max thanks to cudaGetDeviceProperties
		cudaDeviceProp prop;
		HANDLE_ERROR( cudaGetDeviceProperties( &prop, 0 ) ); // Only one GPU so device number is 0
		const unsigned int nbThreads = prop.maxThreadsPerBlock; // 1024 on my GPU
		// Determine the number of blocks necessary : nbBlocks * nbThreads >= size
		const unsigned int nbBlocks = (size + nbThreads - 1) / nbThreads;
		
		// Launch kernel
		std::cout << "Addition on GPU (" << nbBlocks << " blocks - " << nbThreads << " threads)" << std::endl;
		chrGPU.start();
		sumArraysCUDA<<< nbBlocks, nbThreads >>>(size, dev_a, dev_b, dev_res);
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl;

		// Copy data from device to host (output array)   
		std::cout << "Copy output from device to host" << std::endl;
		HANDLE_ERROR( cudaMemcpy( res, dev_res, bytes, cudaMemcpyDeviceToHost ) ); 

		// Free arrays on device
		std::cout << "Free arrays on GPU" << std::endl;
		cudaFree( dev_a );
		cudaFree( dev_b );
		cudaFree( dev_res );
	}
}
