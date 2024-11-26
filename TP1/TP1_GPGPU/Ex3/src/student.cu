/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 3: Filtre d'images sepia
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC
{
	__global__ void sepiaKernel(uchar *input, uchar *output, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // place en x dans le bloc
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // place en y dans le bloc
    int pixelIndex = (idy * width + idx) * 3;  // pixel a 3 composantes r g b

    if (idx < width && idy < height) {
        uchar inRed = input[pixelIndex];
        uchar inGreen = input[pixelIndex + 1];
        uchar inBlue = input[pixelIndex + 2];

        // Appliquer la transformation sépia
        float outRed = fminf(255.0f, (inRed * 0.393f + inGreen * 0.769f + inBlue * 0.189f));
        float outGreen = fminf(255.0f, (inRed * 0.349f + inGreen * 0.686f + inBlue * 0.168f));
        float outBlue = fminf(255.0f, (inRed * 0.272f + inGreen * 0.534f + inBlue * 0.131f));

        // Stocker les résultats dans la mémoire de sortie
        output[pixelIndex] = static_cast<uchar>(outRed);
        output[pixelIndex + 1] = static_cast<uchar>(outGreen);
        output[pixelIndex + 2] = static_cast<uchar>(outBlue);
    }
}

	void studentJob(const std::vector<uchar> &input, const uint width, const uint height, std::vector<uchar> &output)
	{
		ChronoGPU chrGPU;

		// 2 arrays for GPU
		uchar *dev_input = NULL;
		uchar *dev_output = NULL;
		int size = width * height * 3; // Taille totale de l'image (3 canaux par pixel)

		chrGPU.start();
		const size_t bytes = size * sizeof(uchar);
		cudaMalloc((void **) &dev_input, bytes);
		cudaMalloc((void **) &dev_output, bytes);

		cudaMemcpy(dev_input, input.data(), bytes, cudaMemcpyHostToDevice); // copie des pixels

		dim3 blockSize(16, 16);  // Taille du bloc de threads (16x16)
    	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y); // on découpe l'image pour faire bosser les threads

		// fonction ici
		sepiaKernel<<<gridSize, blockSize>>>(dev_input, dev_output, width, height);
		
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		cudaMemcpy(output.data(), dev_output, bytes, cudaMemcpyDeviceToHost);

		cudaFree(dev_input);
		cudaFree(dev_output);
	}
}
