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
	__device__
	float clampf_dev(const float val, const float min , const float max) 
	{
		return fminf(max, fmaxf(min, val));
	}

	__global__
	void conv_naive(const uchar4 *const input, const uint width, const uint height, const float *const mat, const uint matSize, uchar4 *const output) {
		for(uint y = blockIdx.y * blockDim.y + threadIdx.y; y < height; y += gridDim.y * blockDim.y) {
			for(uint x = blockIdx.x * blockDim.x + threadIdx.x; x < width; x += gridDim.x * blockDim.x) {
				float3 sum = make_float3(0.f,0.f,0.f);
				
				// Apply convolution
				for ( uint j = 0; j < matSize; ++j ) 
				{
					for ( uint i = 0; i < matSize; ++i ) 
					{
						int dX = x + i - matSize / 2;
						int dY = y + j - matSize / 2;

						// Handle borders
						if ( dX < 0 ) 
							dX = 0;

						if ( dX >= width ) 
							dX = width - 1;

						if ( dY < 0 ) 
							dY = 0;

						if ( dY >= height ) 
							dY = height - 1;

						const int idMat		= j * matSize + i;
						const int idPixel	= dY * width + dX;
						sum.x += (float)input[idPixel].x * mat[idMat];
						sum.y += (float)input[idPixel].y * mat[idMat];
						sum.z += (float)input[idPixel].z * mat[idMat];
					}
				}
				const int idOut = y * width + x;
				output[idOut].x = (uchar)clampf_dev( sum.x, 0.f, 255.f );
				output[idOut].y = (uchar)clampf_dev( sum.y, 0.f, 255.f );
				output[idOut].z = (uchar)clampf_dev( sum.z, 0.f, 255.f );
				output[idOut].w = 255;
			}
		}
	}
	

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
    void studentJob_Naive(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					)
	{
		std::cout << "====================================================== Naive" << std::endl;

		ChronoGPU chrGPU;

		uchar4 *dev_input = NULL;
		uchar4 *dev_output = NULL;
		float *dev_mat = NULL;

		const size_t bytes = inputImg.size() * sizeof(uchar4);
		const size_t bytes_mat = matConv.size() * sizeof(float);

		std::cout 	<< "Allocating input (2 arrays) and matrix: " 
					<< ( ( 2 * bytes + bytes_mat ) >> 20 ) << " MB on Device" << std::endl;
		chrGPU.start();		
		HANDLE_ERROR( cudaMalloc( (void**)&dev_input, bytes ) );
		HANDLE_ERROR( cudaMalloc( (void**)&dev_output, bytes ) );	
		HANDLE_ERROR( cudaMalloc( (void**)&dev_mat, bytes_mat ) );	
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from host to device (input array) 
		std::cout << "Copy input from host to device" << std::endl;
		chrGPU.start();		
		HANDLE_ERROR( cudaMemcpy( dev_input, inputImg.data(), bytes, cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy( dev_mat, matConv.data(), bytes_mat, cudaMemcpyHostToDevice ) );
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		cudaDeviceProp prop;
		const dim3 threads(32, 32); // 32 * 32 = 1024
		const dim3 blocks( ( imgWidth + threads.x - 1 ) / threads.x, ( imgHeight + threads.y - 1 ) / threads.y );

		std::cout << "-> Size of blocks x : " << blocks.x << " || Size of blocks y : " << blocks.y << std::endl;
		
		// Launch kernel
		std::cout << "Naive Convolution on GPU (" 	<< blocks.x << "x" << blocks.y << " blocks - " 
												<< threads.x << "x" << threads.y << " threads)" << std::endl;
		chrGPU.start();
		conv_naive<<<blocks, threads>>>(dev_input, imgWidth, imgHeight, dev_mat, matSize, dev_output);
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from device to host (output array)   
		std::cout << "Copy output from device to host" << std::endl;
		chrGPU.start();
		HANDLE_ERROR( cudaMemcpy( output.data(), dev_output, bytes, cudaMemcpyDeviceToHost ) ); 
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl;

		// Free arrays on device
		std::cout << "Free memory on GPU" << std::endl;
		cudaFree(dev_input);
		cudaFree(dev_output);
		cudaFree(dev_mat);

		std::cout << "Comparison of final results" << std::endl;
		compareImages(resultCPU, output);

		std::cout << "============================================================" << std::endl << std::endl;
	}
	
    void studentJob_Constant(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					)
	{
		std::cout << "====================================================== Constant" << std::endl;
		
		/// TODO !!!!!!!!!!!!!!!

		std::cout << "Comparison of final results" << std::endl;
		compareImages(resultCPU, output);

		std::cout << "============================================================" << std::endl << std::endl;
	}
	
    void studentJob_Tex1D(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					)
	{
		std::cout << "====================================================== Texture 1D" << std::endl;
		
		/// TODO !!!!!!!!!!!!!!!

		std::cout << "Comparison of final results" << std::endl;
		compareImages(resultCPU, output);

		std::cout << "============================================================" << std::endl << std::endl;
	}
	
    void studentJob_Tex2D(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					)
	{
		std::cout << "====================================================== Texture 2D" << std::endl;
		
		/// TODO !!!!!!!!!!!!!!!

		std::cout << "Comparison of final results" << std::endl;
		compareImages(resultCPU, output);

		std::cout << "============================================================" << std::endl << std::endl;
	}
	
    void studentJob(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					)
	{
		studentJob_Naive(inputImg,imgWidth,imgHeight,matConv,matSize,resultCPU,output);
		//studentJob_Constant(inputImg,imgWidth,imgHeight,matConv,matSize,resultCPU,output);
		//studentJob_Tex1D(inputImg,imgWidth,imgHeight,matConv,matSize,resultCPU,output);
		//studentJob_Tex2D(inputImg,imgWidth,imgHeight,matConv,matSize,resultCPU,output);		
	}
}
