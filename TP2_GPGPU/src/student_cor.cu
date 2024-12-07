/*
* TP 3 - Convolution d'images
* --------------------------
* Mémoire constante et textures
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

#define USE_CONSTANT
#define USE_TEXTURE_1D
#define USE_TEXTURE_2D
#define NB_REPETITIONS_GPU 100

namespace IMAC
{
	// ========================================================================================== COMPARISON
	// For image comparison
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
					std::cout 	<< "Error at index " << i 
								<< " - a = " << a[i] 
								<< " - b = " << b[i] << " - " 
								<< std::abs(a[i].x - b[i].x) << std::endl;
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
	// ======================================================================================================

	// ====================================================================================================== UTILS

	// Device functions can be called from kernel !
	inline __device__ 
    float cudaClampi(const int val, const int i_min, const int i_max)
    {
        return min(i_max, max(i_min, val));
    }
	__device__
    float cudaClampf(const float val, const float min, const float max)
    {
        return fminf(max, fmaxf(min, val));
    }
	// ======================================================================================================

	// ====================================================================================================== NAIVE
	// ====================================================================================================== NAIVE
	// ====================================================================================================== NAIVE
	__global__
    void convCUDA(	const uchar4 *const input, const uint imgWidth, const uint imgHeight, 
					const float *const matConv, const uint matSize,
					uchar4 *const output)
	{
		for(int y = blockIdx.y * blockDim.y + threadIdx.y; y < imgHeight; y += gridDim.y * blockDim.y) 
		{
			for(int x = blockIdx.x * blockDim.x + threadIdx.x; x < imgWidth; x += gridDim.x * blockDim.x) 
			{
				float3 sum = make_float3(0.f,0.f,0.f);
				for (int j = 0; j < matSize; ++j) 
				{
					for (int i = 0; i < matSize; ++i) 
					{
						int dX = x + i - matSize / 2;
						int dY = y + j - matSize / 2;
						if ( dX < 0 ) dX = 0;
						if ( dX >= imgWidth ) dX = imgWidth - 1;
						if ( dY < 0 ) dY = 0;
						if ( dY >= imgHeight ) dY = imgHeight - 1;

						const int idMat	= j * matSize + i;
						const int idPixel	= dY * imgWidth + dX;
						sum.x += (float)input[idPixel].x * matConv[idMat];
						sum.y += (float)input[idPixel].y * matConv[idMat];
						sum.z += (float)input[idPixel].z * matConv[idMat];
					}
				}
				const int idOut = y * imgWidth + x;
				output[idOut].x = (uchar)cudaClampf(sum.x, 0.f, 255.f);
				output[idOut].y = (uchar)cudaClampf(sum.y, 0.f, 255.f);
				output[idOut].z = (uchar)cudaClampf(sum.z, 0.f, 255.f);
				output[idOut].w = 255;
			}
		}

	}
	void doNaive(	const std::vector<uchar4> &input, const uint imgWidth, const uint imgHeight, 
					const std::vector<float> &matConv, const uint matSize,
					const std::vector<uchar4> &resultCPU, // Just for comparison
					std::vector<uchar4> &output)
	{
		std::cout << "====================================================== Naive" << std::endl;
		// 3 arrays for GPU
		uchar4 *dev_input = NULL;
		uchar4 *dev_output = NULL;
		float *dev_matConv = NULL;

		// Allocate arrays on device (input and ouput)
		const size_t bytesImg = input.size() * sizeof(uchar4);
		const size_t bytesConvMat = matConv.size() * sizeof(float);
		std::cout 	<< "Allocating input, output and convolution matrix on GPU" << bytesImg<<std::endl;
		HANDLE_ERROR(cudaMalloc((void**)&dev_input, bytesImg));
		HANDLE_ERROR(cudaMalloc((void**)&dev_output, bytesImg));
		HANDLE_ERROR(cudaMalloc((void**)&dev_matConv, bytesConvMat));

		// Copy data from host to device (input arrays) 
		std::cout << "Copy data from host to device" << std::endl;
		HANDLE_ERROR(cudaMemcpy(dev_input, input.data(), bytesImg, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(dev_matConv, matConv.data(), bytesConvMat, cudaMemcpyHostToDevice));

		// Configure kernel
		const dim3 nbThreads(32, 32);
		const dim3 nbBlocks((imgWidth + nbThreads.x - 1) / nbThreads.x, (imgHeight + nbThreads.y - 1) / nbThreads.y);

		std::cout << "Process on GPU (" << nbBlocks.x << "x" << nbBlocks.y << " blocks - " 
										<< nbThreads.x << "x" << nbThreads.y << " threads)" << std::endl;
								
		ChronoGPU chrGPU;
		float sum = 0;
		int repetitions = NB_REPETITIONS_GPU;
		for (int i = 0; i < repetitions; i ++) {
			chrGPU.start();
			convCUDA<<< nbBlocks, nbThreads >>>(dev_input, imgWidth, imgHeight, dev_matConv, matSize, dev_output);
			chrGPU.stop();
			sum += chrGPU.elapsedTime();
		}
		std::cout << "Done (naive) : " << (sum / repetitions) << "ms to complete" << std::endl;

		std::cout << "Checking result..." << std::endl;
		// Copy data from device to host (output array)   
		HANDLE_ERROR(cudaMemcpy(output.data(), dev_output, bytesImg, cudaMemcpyDeviceToHost)); 

		compareImages(resultCPU, output);

		// Free arrays on device
		cudaFree(dev_input);
		cudaFree(dev_output);
		cudaFree(dev_matConv);
		std::cout << "============================================================" << std::endl << std::endl;
	}

	// ====================================================================================================== END NAIVE


	// ====================================================================================================== CONSTANT	
	// ====================================================================================================== CONSTANT	
	// ====================================================================================================== CONSTANT	
	// Constant memory must be declared in global with static allocation
	__device__ __constant__ float c_matConv[225];

	// We don't need to give matConv as parameter, it is in constant memory !
	__global__
    void convCUDA_const(	const uchar4 *const input, const uint imgWidth, const uint imgHeight, 
							/* const float *const matConv, */ const uint matSize,
							uchar4 *const output)
	{
		for(int y = blockIdx.y * blockDim.y + threadIdx.y; y < imgHeight; y += gridDim.y * blockDim.y) 
		{
			for(int x = blockIdx.x * blockDim.x + threadIdx.x; x < imgWidth; x += gridDim.x * blockDim.x) 
			{
				float3 sum = make_float3(0.f,0.f,0.f);
				for (int j = 0; j < matSize; ++j) 
				{
					for (int i = 0; i < matSize; ++i) 
					{
						int dX = x + i - matSize / 2;
						int dY = y + j - matSize / 2;
						if ( dX < 0 ) dX = 0;
						if ( dX >= imgWidth ) dX = imgWidth - 1;
						if ( dY < 0 ) dY = 0;
						if ( dY >= imgHeight ) dY = imgHeight - 1;

						const int idMat	    = j * matSize + i;
						const int idPixel	= dY * imgWidth + dX;
						sum.x += (float)input[idPixel].x * c_matConv[idMat]; // Access constant memory
						sum.y += (float)input[idPixel].y * c_matConv[idMat];
						sum.z += (float)input[idPixel].z * c_matConv[idMat];
					}
				}
				const int idOut = y * imgWidth + x;
				output[idOut].x = (uchar)cudaClampf(sum.x, 0.f, 255.f);
				output[idOut].y = (uchar)cudaClampf(sum.y, 0.f, 255.f);
				output[idOut].z = (uchar)cudaClampf(sum.z, 0.f, 255.f);
				output[idOut].w = 255;
			}
		}
	}

	void doConstant(	const std::vector<uchar4> &input, const uint imgWidth, const uint imgHeight, 
						const std::vector<float> &matConv, const uint matSize,
						const std::vector<uchar4> &resultCPU, // Just for comparison
						std::vector<uchar4> &output)
	{
		std::cout << "====================================================== Constant" << std::endl;
		// 2 arrays for GPU (matConv in constant)
		uchar4 *dev_input = NULL;
		uchar4 *dev_output = NULL;

		// Allocate arrays on device (input and ouput)
		const size_t bytesImg = input.size() * sizeof(uchar4);
		const size_t bytesConvMat = matConv.size() * sizeof(float);
		std::cout 	<< "Allocating input and output matrix on GPU" << std::endl;
		HANDLE_ERROR(cudaMalloc((void**)&dev_input, bytesImg));
		HANDLE_ERROR(cudaMalloc((void**)&dev_output, bytesImg));

		// Copy data from host to device (input arrays) 
		std::cout << "Copy data from host to device (input in global memory)" << std::endl;
		HANDLE_ERROR(cudaMemcpy(dev_input, input.data(), bytesImg, cudaMemcpyHostToDevice));
		
		// Copy constant memory
		std::cout << "Copy data from host to device (matConv in constant memory)" << std::endl;
		HANDLE_ERROR(cudaMemcpyToSymbol(c_matConv, matConv.data(), bytesConvMat));

		// Configure kernel
		const dim3 nbThreads(32, 32);
		const dim3 nbBlocks((imgWidth + nbThreads.x - 1) / nbThreads.x, (imgHeight + nbThreads.y - 1) / nbThreads.y);

		std::cout << "Process on GPU (" << nbBlocks.x << "x" << nbBlocks.y << " blocks - " 
										<< nbThreads.x << "x" << nbThreads.y << " threads)" << std::endl;
		ChronoGPU chrGPU;
		float sum = 0;
		int repetitions = NB_REPETITIONS_GPU;
		for (int i = 0; i < repetitions; i ++) {
			chrGPU.start();
			convCUDA_const<<< nbBlocks, nbThreads >>>(dev_input, imgWidth, imgHeight, matSize, dev_output);
			chrGPU.stop();
			sum += chrGPU.elapsedTime();
		}
		std::cout << "Done (Constant) : " << (sum / repetitions) << "ms to complete" << std::endl;

		std::cout << "Checking result..." << std::endl;
		// Copy data from device to host (output array)   
		HANDLE_ERROR(cudaMemcpy(output.data(), dev_output, bytesImg, cudaMemcpyDeviceToHost)); 

		compareImages(resultCPU, output);

		// Free arrays on device
		cudaFree(dev_input);
		cudaFree(dev_output);
		std::cout << "============================================================" << std::endl << std::endl;
	}

	// ====================================================================================================== END CONSTANT
	
	// ====================================================================================================== TEXTURE 1D 
	// ====================================================================================================== TEXTURE 1D 
	// ====================================================================================================== TEXTURE 1D 
	// A global 1D texture
	texture<uchar4, cudaTextureType1D, cudaReadModeElementType> t_in1D;
	
	// We don't need to pass input as parameter, it is in a global texture !
	__global__
	void convGPU_texture1D(	/* const uchar4 *const input,*/ const uint imgWidth, const uint imgHeight,
							const uint matSize,
							uchar4 *const output) 
	{
		for(int y = blockIdx.y * blockDim.y + threadIdx.y; y < imgHeight; y += gridDim.y * blockDim.y) 
		{
			for(int x = blockIdx.x * blockDim.x + threadIdx.x; x < imgWidth; x += gridDim.x * blockDim.x) 
			{
				float3 sum = make_float3(0.f,0.f,0.f);
				for (int j = 0; j < matSize; ++j) 
				{
					for (int i = 0; i < matSize; ++i) 
					{
						// Handle borders
						int dX = x + i - matSize / 2;
						int dY = y + j - matSize / 2;
						if ( dX < 0 ) dX = 0;
						if ( dX >= imgWidth ) dX = imgWidth - 1;
						if ( dY < 0 ) dY = 0;
						if ( dY >= imgHeight ) dY = imgHeight - 1;

						const int idMat	    = j * matSize + i;
						const int idPixel	= dY * imgWidth + dX;
						uchar4 in = tex1Dfetch(t_in1D, idPixel); // Get data from 1D texture
						sum.x += (float)in.x * c_matConv[idMat];
						sum.y += (float)in.y * c_matConv[idMat];
						sum.z += (float)in.z * c_matConv[idMat];
					}
				}
				const int idOut = y * imgWidth + x;
				output[idOut].x = (uchar)cudaClampf(sum.x, 0.f, 255.f);
				output[idOut].y = (uchar)cudaClampf(sum.y, 0.f, 255.f);
				output[idOut].z = (uchar)cudaClampf(sum.z, 0.f, 255.f);
				output[idOut].w = 255;
			}
		}
	}

	void doTexture1D(	const std::vector<uchar4> &input, const uint imgWidth, const uint imgHeight, 
						const std::vector<float> &matConv, const uint matSize,
						const std::vector<uchar4> &resultCPU, // Just for comparison
						std::vector<uchar4> &output)
	{
		std::cout << "====================================================== Texture 1D" << std::endl;
		// 2 arrays for GPU (matConv in constant)
		uchar4 *dev_input = NULL;
		uchar4 *dev_output = NULL;

		// Allocate arrays on device (input and ouput)
		const size_t bytesImg = input.size() * sizeof(uchar4);
		const size_t bytesConvMat = matConv.size() * sizeof(float);
		std::cout 	<< "Allocating input and output on GPU" << std::endl;
		HANDLE_ERROR(cudaMalloc((void**)&dev_input, bytesImg));
		HANDLE_ERROR(cudaMalloc((void**)&dev_output, bytesImg));

		// Copy data from host to device (input arrays) 
		std::cout << "Copy data from host to device (input in global memory)" << std::endl;
		HANDLE_ERROR(cudaMemcpy(dev_input, input.data(), bytesImg, cudaMemcpyHostToDevice));
		// Bind 1D texture
		std::cout << "Bind input on texture 1D" << std::endl;
		HANDLE_ERROR(cudaBindTexture(NULL, t_in1D, dev_input, bytesImg));
		
		// Copy constant memory
		std::cout << "Copy data from host to device (matConv in constant memory)" << std::endl;
		HANDLE_ERROR(cudaMemcpyToSymbol(c_matConv, matConv.data(), bytesConvMat));

		// Configure kernel
		const dim3 nbThreads(32, 32);
		const dim3 nbBlocks((imgWidth + nbThreads.x - 1) / nbThreads.x, (imgHeight + nbThreads.y - 1) / nbThreads.y);

		std::cout << "Process on GPU (" << nbBlocks.x << "x" << nbBlocks.y << " blocks - " 
										<< nbThreads.x << "x" << nbThreads.y << " threads)" << std::endl;
								
		ChronoGPU chrGPU;
		float sum = 0;
		int repetitions = NB_REPETITIONS_GPU;
		for (int i = 0; i < repetitions; i ++) {
			chrGPU.start();
			convGPU_texture1D<<< nbBlocks, nbThreads >>>(imgWidth, imgHeight, matSize, dev_output);
			chrGPU.stop();
			sum += chrGPU.elapsedTime();
		}
		std::cout << "Done (Texture 1D) : " << (sum / repetitions) << "ms to complete" << std::endl;

		std::cout << "Checking result..." << std::endl;
		// Copy data from device to host (output array)   
		HANDLE_ERROR(cudaMemcpy(output.data(), dev_output, bytesImg, cudaMemcpyDeviceToHost)); 

		compareImages(resultCPU, output);

		// Unbind texture
		HANDLE_ERROR(cudaUnbindTexture(t_in1D));
		
		// Free arrays on device
		cudaFree(dev_input);
		cudaFree(dev_output);
		std::cout << "============================================================" << std::endl << std::endl;
	}
	// ====================================================================================================== END TEXTURE 1D

	
	// ====================================================================================================== TEXTURE 1D NEW
	// We don't need to pass input as parameter, it is in a global texture !
	__global__
	void convGPU_texture1D_New(const uint imgWidth, const uint imgHeight,
								const uint matSize,
								uchar4 *const output, cudaTextureObject_t tex_in1D) 
	{
		for(int y = blockIdx.y * blockDim.y + threadIdx.y; y < imgHeight; y += gridDim.y * blockDim.y) 
		{
			for(int x = blockIdx.x * blockDim.x + threadIdx.x; x < imgWidth; x += gridDim.x * blockDim.x) 
			{
				float3 sum = make_float3(0.f,0.f,0.f);
				for (int j = 0; j < matSize; ++j) 
				{
					for (int i = 0; i < matSize; ++i) 
					{
						// Handle borders
						int dX = x + i - matSize / 2;
						int dY = y + j - matSize / 2;
						if ( dX < 0 ) dX = 0;
						if ( dX >= imgWidth ) dX = imgWidth - 1;
						if ( dY < 0 ) dY = 0;
						if ( dY >= imgHeight ) dY = imgHeight - 1;

						const int idMat	    = j * matSize + i;
						const int idPixel	= dY * imgWidth + dX;
						uchar4 in = tex1Dfetch<uchar4>(tex_in1D, idPixel); // Get data from 1D texture
						sum.x += (float)in.x * c_matConv[idMat];
						sum.y += (float)in.y * c_matConv[idMat];
						sum.z += (float)in.z * c_matConv[idMat];
					}
				}
				const int idOut = y * imgWidth + x;
				output[idOut].x = (uchar)cudaClampf(sum.x, 0.f, 255.f);
				output[idOut].y = (uchar)cudaClampf(sum.y, 0.f, 255.f);
				output[idOut].z = (uchar)cudaClampf(sum.z, 0.f, 255.f);
				output[idOut].w = 255;
			}
		}
	}

	void doTexture1DNew(const std::vector<uchar4> &input, const uint imgWidth, const uint imgHeight, 
						const std::vector<float> &matConv, const uint matSize,
						const std::vector<uchar4> &resultCPU, // Just for comparison
						std::vector<uchar4> &output)
	{
		std::cout << "====================================================== Texture 1D Un Deprecated" << std::endl;
		// 2 arrays for GPU (matConv in constant)
		uchar4 *dev_input = NULL;
		uchar4 *dev_output = NULL;

		cudaTextureObject_t tex1Din;

		// Allocate arrays on device (input and ouput)
		const size_t bytesImg = input.size() * sizeof(uchar4);
		const size_t bytesConvMat = matConv.size() * sizeof(float);
		std::cout 	<< "Allocating input and output on GPU" << std::endl;
		HANDLE_ERROR(cudaMalloc((void**)&dev_input, bytesImg));
		HANDLE_ERROR(cudaMalloc((void**)&dev_output, bytesImg));

		// Copy data from host to device (input arrays) 
		std::cout << "Copy data from host to device (input in global memory)" << std::endl;
		HANDLE_ERROR(cudaMemcpy(dev_input, input.data(), bytesImg, cudaMemcpyHostToDevice));

		// Creation of two structures for the texture object
		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeLinear;
		resDesc.res.linear.devPtr = dev_input;
		resDesc.res.linear.desc = cudaCreateChannelDesc<uchar4>();
		resDesc.res.linear.sizeInBytes = bytesImg;//imgWidth*sizeof(uchar4);
		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.filterMode = cudaFilterModePoint; // Point filter (no linear)
		cudaCreateTextureObject(&tex1Din, &resDesc, &texDesc, NULL);
		
		// Copy constant memory
		std::cout << "Copy data from host to device (matConv in constant memory)" << std::endl;
		HANDLE_ERROR(cudaMemcpyToSymbol(c_matConv, matConv.data(), bytesConvMat));

		// Configure kernel
		const dim3 nbThreads(32, 32);
		const dim3 nbBlocks((imgWidth + nbThreads.x - 1) / nbThreads.x, (imgHeight + nbThreads.y - 1) / nbThreads.y);

		std::cout << "Process on GPU (" << nbBlocks.x << "x" << nbBlocks.y << " blocks - " 
										<< nbThreads.x << "x" << nbThreads.y << " threads)" << std::endl;
								
		ChronoGPU chrGPU;
		float sum = 0;
		int repetitions = NB_REPETITIONS_GPU;
		for (int i = 0; i < repetitions; i ++) {
			chrGPU.start();
			convGPU_texture1D_New<<< nbBlocks, nbThreads >>>(imgWidth, imgHeight, matSize, dev_output, tex1Din);
			chrGPU.stop();
			sum += chrGPU.elapsedTime();
		}
		std::cout << "Done (Texture 1D) : " << (sum / repetitions) << "ms to complete" << std::endl;

		std::cout << "Checking result..." << std::endl;
		// Copy data from device to host (output array)   
		HANDLE_ERROR(cudaMemcpy(output.data(), dev_output, bytesImg, cudaMemcpyDeviceToHost)); 

		compareImages(resultCPU, output);

		// Free arrays on device
		cudaFree(dev_input);
		cudaFree(dev_output);
		std::cout << "============================================================" << std::endl << std::endl;
	}


	// ====================================================================================================== TEXTURE 2D 

	// A global 2D texture
	//texture<uchar4, cudaTextureType2D, cudaReadModeElementType> t_in2D;

	// Same as before...
	__global__
	void convGPU_texture2D(	const uint imgWidth, const uint imgHeight,
							const uint matSize,uchar4 *const output, cudaTextureObject_t texin) 
	{
		int x,y,i,j,dX,dY,idMat,idOut;
		for(y = blockIdx.y * blockDim.y + threadIdx.y; y < imgHeight; y += gridDim.y * blockDim.y) 
		{
			for(x = blockIdx.x * blockDim.x + threadIdx.x; x < imgWidth; x += gridDim.x * blockDim.x) 
			{
				//if (x < imgWidth && y < imgHeight){
				float3 sum = {0,0,0};
				for (i = 0; i < matSize; ++i) 
				{
					for (j = 0; j < matSize; ++j) 
					{
						/*
						int dX = cudaClampi(x + i - matSize / 2, 0, imgWidth - 1);
						int dY = cudaClampi(y + j - matSize / 2, 0, imgHeight - 1);
						*/
						// Handle borders
						dX = x + i - matSize / 2;
						dY = y + j - matSize / 2;
						if ( dX < 0 ) dX = 0;
						if ( dX >= imgWidth ) dX = imgWidth - 1;
						if ( dY < 0 ) dY = 0;
						if ( dY >= imgHeight ) dY = imgHeight - 1;
						
						idMat = j * matSize + i;

						//const uint idMat = j * matSize + i;
						uchar4 in = tex2D<uchar4>(texin, dX, dY); // Get data from 2D texture
						sum.x += (float)in.x * c_matConv[idMat];
						sum.y += (float)in.y * c_matConv[idMat];
						sum.z += (float)in.z * c_matConv[idMat];
					}
				}
				idOut = y * imgWidth + x;
				output[idOut].x = (uchar)cudaClampf(sum.x, 0.f, 255.f);
				output[idOut].y = (uchar)cudaClampf(sum.y, 0.f, 255.f);
				output[idOut].z = (uchar)cudaClampf(sum.z, 0.f, 255.f);
				output[idOut].w = 255;
			}
		}
	}


	void doTexture2D(	const std::vector<uchar4> &input, const uint imgWidth, const uint imgHeight, 
						const std::vector<float> &matConv, const uint matSize,
						const std::vector<uchar4> &resultCPU, // Just for comparison
						std::vector<uchar4> &output)
	{
		std::cout << "====================================================== Texture 2D" << std::endl;

		cudaTextureObject_t texin;
		ChronoGPU chrGPU;
		uchar4 *dev_input = NULL;
		uchar4 *dev_output = NULL;

		//size_t pitch;
		//const size_t widthBytes = imgWidth * sizeof(uchar4);
		const size_t imgBytes = input.size() * sizeof(uchar4);
		const size_t matBytes = matConv.size() * sizeof(float);

		const size_t bytesImgWidth = imgWidth * sizeof(uchar4);

		size_t inputPitch;
		std::cout 	<< "Allocating input and ouput (2 arrays): "  << ( ( 2 * imgBytes ) >> 20 ) << " MB on Device" << std::endl;
		// Allocate arrays on device (input, ouput and matrix)
		/* VERSION AVEC PITCH */
		HANDLE_ERROR(cudaMallocPitch((void**)&dev_input, &inputPitch, bytesImgWidth, imgHeight));
		HANDLE_ERROR(cudaMemcpy2D(	dev_input, inputPitch, input.data(),bytesImgWidth, bytesImgWidth, imgHeight, cudaMemcpyHostToDevice));
		/* VERSION SANS PITCH */
		// HANDLE_ERROR( cudaMalloc((void **) &dev_input,   imgBytes) );
		// HANDLE_ERROR( cudaMemcpy(dev_input, input.data(), imgBytes, cudaMemcpyHostToDevice) );
		// inputPitch = imgWidth*sizeof(uchar4);

		HANDLE_ERROR( cudaMalloc((void **) &dev_output,   imgBytes) );

		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypePitch2D;
		resDesc.res.pitch2D.devPtr = dev_input;
		resDesc.res.pitch2D.width = imgWidth;
		resDesc.res.pitch2D.height = imgHeight;
		resDesc.res.pitch2D.desc = cudaCreateChannelDesc<uchar4>();
		resDesc.res.pitch2D.pitchInBytes = inputPitch;
		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.filterMode = cudaFilterModePoint; // Point filter (no linear)
		cudaCreateTextureObject(&texin, &resDesc, &texDesc, NULL);
		
		// Copy data from host to device (input arrays)
		std::cout << "Copy input from host to device" << std::endl;
		HANDLE_ERROR( cudaMemcpyToSymbol(c_matConv, matConv.data(), matBytes, 0, cudaMemcpyHostToDevice) );

		// Bind texture 2D
		// No binding with texture object

		// Launch kernel
		const dim3 nb_threads(32, 32);
		const dim3 nb_blocks((imgWidth + nb_threads.x - 1) / nb_threads.x, (imgHeight + nb_threads.y - 1) / nb_threads.y);
		std::cout << "Process on GPU (" 	<< nb_blocks.x << "x" << nb_blocks.y << " blocks - " << nb_threads.x << "x" << nb_threads.y << " threads)" << std::endl;
		chrGPU.start();
		convGPU_texture2D<<<nb_blocks, nb_threads>>>(imgWidth, imgHeight, matSize, dev_output, texin);
		chrGPU.stop();
		std::cout 	<< "Done (Texture 2D) : " << chrGPU.elapsedTime() << " ms to complete" << std::endl;

		// Copy data from device to host (output array)
		std::cout << "Copy output from device to host" << std::endl;
		HANDLE_ERROR( cudaMemcpy(output.data(), dev_output, imgBytes, cudaMemcpyDeviceToHost) );
		
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		std::cout << "Comparison of final results" << std::endl;
		compareImages(resultCPU, output);

		// Free array on device
		cudaFree(dev_output);
		cudaFree(dev_input);
		std::cout << "============================================================" << std::endl << std::endl;
	}

// ====================================================================================================== STUDENT JOB
    void studentJob(const std::vector<uchar4> &input, const uint imgWidth, const uint imgHeight, 
                    const std::vector<float> &matConv, const uint matSize,
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output)
	{
		doNaive(input, imgWidth, imgHeight, matConv, matSize, resultCPU, output);
		doConstant(input, imgWidth, imgHeight, matConv, matSize, resultCPU, output);
		doTexture1D(input, imgWidth, imgHeight, matConv, matSize, resultCPU, output);
		doTexture1DNew(input, imgWidth, imgHeight, matConv, matSize, resultCPU, output);
		doTexture2D(input, imgWidth, imgHeight, matConv, matSize, resultCPU, output);
	}
}
