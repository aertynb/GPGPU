/*
* TP 3 - Réduction CUDA
* --------------------------
* Mémoire paratagée, synchronisation, optimisation
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"

namespace IMAC
{
	// ==================================================== EX 1
    __global__
    void maxReduce_ex1(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{	
		extern __shared__ uint values[];
		int idX = blockIdx.x * blockDim.x + threadIdx.x; int index = threadIdx.x;
		
		if (idX >= size) {
			return;
		}

		values[index] = dev_array[idX];
		__syncthreads();
		

		for(int i = 1; i < blockDim.x; i *= 2) {
			if (index + i < blockDim.x) {
				values[index] = umax(values[index], values[index + i]);
				
			}	
			__syncthreads();
		}
		
		if (threadIdx.x == 0)
			dev_partialMax[blockIdx.x] = values[index];
	}

	__global__
    void maxReduce_ex2(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ uint values[];
		int idX = blockIdx.x * blockDim.x + threadIdx.x; int index = threadIdx.x;
		
		if (idX >= size) {
			return;
		}

		values[index] = dev_array[idX];
		__syncthreads();
		

		for(int i = blockDim.x / 2; i > 0; i /= 2) {
			if (index + i < blockDim.x) {
				values[index] = umax(values[index], values[index + i]);
				
			}	
			__syncthreads();
		}
		
		if (threadIdx.x == 0)
			dev_partialMax[blockIdx.x] = values[index];
	}

	__global__
	void maxReduce_ex3(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ uint values[];
		int globalIndex = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
		int localIndex = threadIdx.x;

		uint val1 = (globalIndex < size) ? dev_array[globalIndex] : 0;
		uint val2 = (globalIndex + 1 < size) ? dev_array[globalIndex + 1] : 0;

		values[localIndex] = umax(val1, val2);
		__syncthreads();

		for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
			if (localIndex < stride) {
				values[localIndex] = umax(values[localIndex], values[localIndex + stride]);
			}
			__syncthreads();
		}

		if (localIndex == 0) {
			dev_partialMax[blockIdx.x] = values[0];
		}
	}

	__global__
void maxReduce_ex4(const uint *const dev_array, const uint size, uint *const dev_partialMax)
{
    extern __shared__ uint values[];
    int globalIndex = 2 * (blockIdx.x * blockDim.x + threadIdx.x); 
    int localIndex = threadIdx.x;

    uint val1 = (globalIndex < size) ? dev_array[globalIndex] : 0;
    uint val2 = (globalIndex + 1 < size) ? dev_array[globalIndex + 1] : 0;

    values[localIndex] = umax(val1, val2);
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride /= 2) {
        if (localIndex < stride) {
            values[localIndex] = umax(values[localIndex], values[localIndex + stride]);
        }
        __syncthreads();
    }

    volatile uint* vvalues = values;

    if (localIndex < 32) {
        vvalues[localIndex] = umax(vvalues[localIndex], vvalues[localIndex + 32]);
        vvalues[localIndex] = umax(vvalues[localIndex], vvalues[localIndex + 16]);
        vvalues[localIndex] = umax(vvalues[localIndex], vvalues[localIndex + 8]);
        vvalues[localIndex] = umax(vvalues[localIndex], vvalues[localIndex + 4]);
        vvalues[localIndex] = umax(vvalues[localIndex], vvalues[localIndex + 2]);
        vvalues[localIndex] = umax(vvalues[localIndex], vvalues[localIndex + 1]);
    }

    // Le thread 0 écrit le maximum partiel dans la mémoire globale
    if (localIndex == 0) {
        dev_partialMax[blockIdx.x] = values[0];
    }
}

	void studentJob(const std::vector<uint> &array, const uint resCPU /* Just for comparison */, const uint nbIterations)
    {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
		
		uint *dev_array = NULL;
        const size_t bytes = array.size() * sizeof(uint);

		// Allocate array on GPU
		HANDLE_ERROR( cudaMalloc( (void**)&dev_array, bytes ) );
		// Copy data from host to device
		HANDLE_ERROR( cudaMemcpy( dev_array, array.data(), bytes, cudaMemcpyHostToDevice ) );

		std::cout << "Test with " << nbIterations << " iterations" << std::endl;

		std::cout << "========== Ex 1 " << std::endl;
		uint res1 = 0; // result
		// Launch reduction and get timing
		float2 timing1 = reduce<KERNEL_EX1>(nbIterations, dev_array, array.size(), res1);
		
        std::cout << " -> Done: ";
        printTiming(timing1);
		compare(res1, resCPU); // Compare results

		std::cout << "========== Ex 2 " << std::endl;
		uint res2 = 0; // result
		// Launch reduction and get timing
		float2 timing2 = reduce<KERNEL_EX2>(nbIterations, dev_array, array.size(), res2);
		
        std::cout << " -> Done: ";
        printTiming(timing2);
		compare(res2, resCPU);

		std::cout << "========== Ex 3 " << std::endl;
		uint res3 = 0; // result
		// Launch reduction and get timing
		float2 timing3 = reduce<KERNEL_EX3>(nbIterations, dev_array, array.size(), res3);
		
        std::cout << " -> Done: ";
        printTiming(timing3);
		compare(res3, resCPU);

		std::cout << "========== Ex 4 " << std::endl;
		uint res4 = 0; // result
		// Launch reduction and get timing
		float2 timing4 = reduce<KERNEL_EX4>(nbIterations, dev_array, array.size(), res4);
		
        std::cout << " -> Done: ";
        printTiming(timing4);
		compare(res4, resCPU);

		std::cout << "========== Ex 5 " << std::endl;
		uint res5 = 0; // result
		// Launch reduction and get timing
		float2 timing5 = reduce<KERNEL_EX5>(nbIterations, dev_array, array.size(), res5);
		
        std::cout << " -> Done: ";
        printTiming(timing5);
		compare(res5, resCPU);

		// Free array on GPU
		cudaFree( dev_array );
    }

	void printTiming(const float2 timing)
	{
		std::cout << ( timing.x < 1.f ? 1e3f * timing.x : timing.x ) << " us on device and ";
		std::cout << ( timing.y < 1.f ? 1e3f * timing.y : timing.y ) << " us on host." << std::endl;
	}

    void compare(const uint resGPU, const uint resCPU)
	{
		if (resGPU == resCPU)
		{
			std::cout << "Well done ! " << resGPU << " == " << resCPU << " !!!" << std::endl;
		}
		else
		{
			std::cout << "You failed ! " << resGPU << " != " << resCPU << " !!!" << std::endl;
		}
	}
}
