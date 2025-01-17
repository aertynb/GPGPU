/*
* TP 2 - Convolution d'images
* --------------------------
* Mémoire constante et textures
*
* File: main.cu
* Author: Maxime MARIA
*/

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <iomanip>     
#include <cstring>
#include <exception>
#include <algorithm>

#include "student.hpp"
#include "chronoCPU.hpp"
#include "lodepng.h"
#include "conv_utils.hpp"

namespace IMAC
{
	// Print program usage
	void printUsageAndExit(const char *prg) 
	{
		std::cerr	<< "Usage: " << prg << std::endl
					<< " \t -f <F>: <F> image file name (required)" 
					<< std::endl;
		exit(EXIT_FAILURE);
	}

	float clampf(const float val, const float min , const float max) 
	{
		return std::min<float>(max, std::max<float>(min, val));
	}
	
	void convCPU(	const std::vector<uchar4> &input, const uint imgWidth, const uint imgHeight, 
					const std::vector<float> &matConv, const uint matSize, 
					std::vector<uchar4> &output)
	{
		std::cout << "Process on CPU (sequential)"	<< std::endl;
		ChronoCPU chrCPU;
		chrCPU.start();
		for ( uint y = 0; y < imgHeight; ++y )
		{
			for ( uint x = 0; x < imgWidth; ++x ) 
			{
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

						if ( dX >= imgWidth ) 
							dX = imgWidth - 1;

						if ( dY < 0 ) 
							dY = 0;

						if ( dY >= imgHeight ) 
							dY = imgHeight - 1;

						const int idMat		= j * matSize + i;
						const int idPixel	= dY * imgWidth + dX;
						sum.x += (float)input[idPixel].x * matConv[idMat];
						sum.y += (float)input[idPixel].y * matConv[idMat];
						sum.z += (float)input[idPixel].z * matConv[idMat];
					}
				}
				const int idOut = y * imgWidth + x;
				output[idOut].x = (uchar)clampf( sum.x, 0.f, 255.f );
				output[idOut].y = (uchar)clampf( sum.y, 0.f, 255.f );
				output[idOut].z = (uchar)clampf( sum.z, 0.f, 255.f );
				output[idOut].w = 255;
			}
		}
		chrCPU.stop();
		std::cout 	<< " -> Done : " << chrCPU.elapsedTime() << " ms" << std::endl << std::endl;
	}

	void RGBtoHSV(const std::vector<uchar4>& input, std::vector<uchar4>& output) {
		for (size_t i = 0; i < input.size(); ++i) {
			// Normalize RGB values to [0, 1]
			float r = input[i].x / 255.0f;
			float g = input[i].y / 255.0f;
			float b = input[i].z / 255.0f;

			float max = fmaxf(fmaxf(r, g), b);
			float min = fminf(fminf(r, g), b);
			float delta = max - min;

			float h = 0.0f;
			float s = 0.0f;
			float v = max;

			// Calculate Hue
			if (delta != 0) {
				if (max == r) {
					h = (g - b) / delta;
				} else if (max == g) {
					h = (b - r) / delta + 2;
				} else {
					h = (r - g) / delta + 4;
				}
				h = (h < 0 ? h + 6 : h) * 60; // Convert to degrees
			}

			// Calculate Saturation
			if (max != 0) {
				s = delta / max;
			}

			// Scale Hue, Saturation, and Value to [0, 255]
			output[i].x = static_cast<unsigned char>(h / 360.0f * 255.0f);
			output[i].y = static_cast<unsigned char>(s * 255.0f);
			output[i].z = static_cast<unsigned char>(v * 255.0f);
			output[i].w = 255;
		}
	}

	void HSVtoRGB(const std::vector<uchar4>& input, std::vector<uchar4>& output) {
		for (size_t i = 0; i < input.size(); ++i) {
			// Normalize HSV values
			float H = input[i].x / 255.0f * 360.0f; // [0, 255] -> [0, 360]
			float S = input[i].y / 255.0f;          // [0, 255] -> [0, 1]
			float V = input[i].z / 255.0f;          // [0, 255] -> [0, 1]

			float C = V * S; // Chroma
			float X = C * (1 - fabsf(fmodf(H / 60.0f, 2) - 1));
			float m = V - C;

			float r = 0, g = 0, b = 0;
			if (H >= 0 && H < 60) {
				r = C; g = X; b = 0;
			} else if (H >= 60 && H < 120) {
				r = X; g = C; b = 0;
			} else if (H >= 120 && H < 180) {
				r = 0; g = C; b = X;
			} else if (H >= 180 && H < 240) {
				r = 0; g = X; b = C;
			} else if (H >= 240 && H < 300) {
				r = X; g = 0; b = C;
			} else if (H >= 300 && H < 360) {
				r = C; g = 0; b = X;
			}

			// Convert back to [0, 255]
			output[i].x = static_cast<unsigned char>((r + m) * 255.0f);
			output[i].y = static_cast<unsigned char>((g + m) * 255.0f);
			output[i].z = static_cast<unsigned char>((b + m) * 255.0f);
			output[i].w = 255;
		}
	}


	const std::vector<int> getCumulHisto(const std::vector<uchar4>& input) {
		int L = 256; 
		int n = input.size();

		// Calculate the histogram
		std::vector<int> histogram(L, 0);
		for (const auto& pixel : input) {
			histogram[pixel.z]++; // Count intensity values (pixel.z is V)
		}

		// Calculate the cumulative histogram
		std::vector<int> cumulative(L, 0);
		cumulative[0] = histogram[0];
		for (int i = 1; i < L; ++i) {
			cumulative[i] = cumulative[i - 1] + histogram[i];
		}
		return cumulative;
	}

	void equalizeHistogramCPU(const std::vector<uchar4>& inputHSV, 
                          std::vector<uchar4>& outputHSV, 
                          const uint imgWidth, 
                          const uint imgHeight) {
		const int L = 256; // Number of intensity levels
		const int numPixels = imgWidth * imgHeight;
		auto histo = getCumulHisto(inputHSV);

		outputHSV.resize(inputHSV.size());
		for (size_t i = 0; i < inputHSV.size(); ++i) {
			outputHSV[i] = inputHSV[i]; // Copy H and S
			outputHSV[i].z = static_cast<uchar>(
					(((L - 1) * static_cast<float>(histo[outputHSV[i].z])) / static_cast<float>(L * numPixels)) * 255);
		}
	}

	// Main function
	void main(int argc, char **argv) 
	{	
		char fileName[2048];
		// Parse command line
		if (argc != 3) 
		{
			std::cerr << "Wrong number of argument" << std::endl;
			printUsageAndExit(argv[0]);
		}

		for (int i = 1; i < argc; ++i) 
		{
			if (!strcmp(argv[i], "-f")) 
			{
				if (sscanf(argv[++i], "%s", fileName) != 1)
				{
					std::cerr << "No file provided after -f" << std::endl;
					printUsageAndExit(argv[0]);
				}
			}
			else
			{
				printUsageAndExit(argv[0]);
			}
		}
		
		// Get input image
		std::vector<uchar> inputUchar;
		uint imgWidth;
		uint imgHeight;

		std::cout << "Loading " << fileName << std::endl;
		unsigned error = lodepng::decode(inputUchar, imgWidth, imgHeight, fileName, LCT_RGBA);
		if (error) {
			throw std::runtime_error("Error lodepng::decode: " + std::string(lodepng_error_text(error)));
		}

		// Convert input to uchar4
		std::vector<uchar4> input;
		input.resize(inputUchar.size() / 4);
		for (uint i = 0; i < input.size(); ++i) {
			const uint id = 4 * i;
			input[i].x = inputUchar[id];     // R
			input[i].y = inputUchar[id + 1]; // G
			input[i].z = inputUchar[id + 2]; // B
			input[i].w = inputUchar[id + 3]; // A
		}

		std::cout << "Image has " << imgWidth << " x " << imgHeight << " pixels (RGBA)" << std::endl;

		// RGB to HSV
		std::vector<uchar4> inputHSV(input.size());
		RGBtoHSV(input, inputHSV);
		
		// Get cumulative histo with CPU
		const auto histoCPU = getCumulHisto(inputHSV);
		std::vector<int> histo (256, 0);

		std::vector<uchar4> outputGPU (imgHeight * imgWidth);
		std::vector<uchar4> outputCPU (imgHeight * imgWidth);

		equalizeHistogramCPU(inputHSV, outputCPU, imgWidth, imgHeight);

		//To do GPU operation
		studentJob(input, imgWidth, imgHeight, histo, histoCPU, outputCPU, outputGPU);

		// HSV to RGB
		std::vector<uchar4> outputRGBCPU(outputGPU.size());
		HSVtoRGB(outputCPU, outputRGBCPU);


		// Prepare output file name
		const std::string fileNameStr(fileName);
		std::size_t lastPoint = fileNameStr.find_last_of(".");
		std::string ext = fileNameStr.substr(lastPoint);
		std::string name = fileNameStr.substr(0,lastPoint);
		std::string outputGPUName = name + "_GPU" + ext;
		std::string outputCPUName = name + "_CPU" + ext;

		std::cout << "Save image as: " << outputGPUName << std::endl;
		error = lodepng::encode(outputGPUName, reinterpret_cast<uchar *>(outputGPU.data()), imgWidth, imgHeight, LCT_RGBA);
		if (error) {
			throw std::runtime_error("Error lodepng::encode: " + std::string(lodepng_error_text(error)));
		}
		error = lodepng::encode(outputCPUName, reinterpret_cast<uchar *>(outputRGBCPU.data()), imgWidth, imgHeight, LCT_RGBA);
		if (error)
		{
			throw std::runtime_error("Error loadpng::decode: " + std::string(lodepng_error_text(error)));
		}
		
		std::cout << "============================================"	<< std::endl << std::endl;
	}
}

int main(int argc, char **argv) 
{
	try
	{
		IMAC::main(argc, argv);
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}
}
