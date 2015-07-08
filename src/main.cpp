#include <vector>
#include <string>

#include <boost/random/mersenne_twister.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <include/HogDescriptor.h>
#include <include/HogGPU.h>

#include "utils.h"
#include "timing.h"
#include "difference.h"

int main(int argc,
	char** argv)
{
	setenv("DISPLAY", ":0", 0);
	std::vector< std::string > file_list = getImagesList("../resources/images");
	ghog::lib::HogDescriptor* hog_cpu = new ghog::lib::HogDescriptor("hog.xml");
	ghog::lib::HogDescriptor* hog_gpu = new ghog::lib::HogGPU("hog.xml");
	cv::Size img_size(3648, 2736);
	cv::Size window_size(1280, 720);
	int num_experiments_timing = 500;
	int num_experiments_difference = 1000;

	boost::random::mt19937 random_gen;

//	measure_difference(hog_cpu, hog_gpu, "Hog CPU", "Hog GPU", file_list,
//		img_size, window_size, num_experiments_difference, random_gen);
//	measure_time_opencv(file_list, img_size, num_experiments_timing,
//		random_gen);
	measure_time(hog_cpu, "My Hog_CPU", file_list, img_size,
		num_experiments_timing, random_gen);
	measure_time(hog_gpu, "My Hog_GPU", file_list, img_size,
		num_experiments_timing, random_gen);

	return 0;
}

