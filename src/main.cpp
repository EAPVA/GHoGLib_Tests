#include <iostream>
#include <vector>
#include <string>

#include <boost/chrono.hpp>
#include <boost/thread.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_smallint.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <include/HogCPU.h>
#include <include/HogGPU.h>

#include "utils.h"

void report_statistics(std::vector< double > values,
	std::string unit)
{
	double max = 0;
	double min = DBL_MAX;
	double average = 0;
	double std_dev = 0;
	double total = 0;

	for(int i = 0; i < values.size(); ++i)
	{
		if(values[i] > max)
		{
			max = values[i];
		}
		if(values[i] < min)
		{
			min = values[i];
		}
		total += values[i];
	}
	average = total / values.size();
	for(int i = 0; i < values.size(); ++i)
	{
		std_dev += (values[i] - average) * (values[i] - average);
	}
	std_dev /= (values.size() - 1);
	std_dev = sqrt(std_dev);

	std::cout << "Average: " << average << " " << unit << std::endl;
	std::cout << "Standard deviation: " << std_dev << " " << unit << std::endl;
	std::cout << "Max: " << max << " " << unit << std::endl;
	std::cout << "Min: " << min << " " << unit << std::endl;
	std::cout << "Total: " << total << " " << unit << std::endl;
	std::cout << std::endl << std::endl;
}

void measure_time(ghog::lib::IHog* hog,
	std::string hog_name,
	std::vector< std::string > image_list,
	cv::Size img_size,
	cv::Size window_size,
	int num_experiments,
	boost::random::mt19937 random_gen)
{
	std::cout << "Running experiment on descriptor " << hog_name << ", with "
		<< image_list.size() << " images, using " << num_experiments
		<< " windows of size " << window_size << std::endl;

	cv::Mat input_img;
	cv::Mat grad_mag;
	cv::Mat grad_phase;

	hog->alloc_buffer(img_size, CV_32FC1, input_img);
	hog->alloc_buffer(window_size, CV_32FC1, grad_mag);
	hog->alloc_buffer(window_size, CV_32FC1, grad_phase);

	boost::chrono::steady_clock::time_point start;
	boost::chrono::duration< double > time_elapsed;

	boost::random::uniform_smallint< int > dist_w(1,
		input_img.cols - window_size.width - 2);
	boost::random::uniform_smallint< int > dist_h(1,
		input_img.rows - window_size.height - 2);

	std::vector< double > values;

	for(int i = 0; i < image_list.size(); ++i)
	{
		cv::imread(image_list[i], CV_LOAD_IMAGE_GRAYSCALE).convertTo(input_img,
			CV_32FC1);

		for(int j = 0; j < num_experiments; ++j)
		{
			int pos_x = dist_w(random_gen);
			int pos_y = dist_h(random_gen);

			start = boost::chrono::steady_clock::now();
			hog->calc_gradient_sync(
				input_img.rowRange(pos_y, pos_y + window_size.height + 1)
					.colRange(pos_x, pos_x + window_size.width + 1), grad_mag,
				grad_phase);
			time_elapsed = boost::chrono::steady_clock::now() - start;
			values.push_back(time_elapsed.count());
		}
	}

	report_statistics(values, "seconds");
}

int main(int argc,
	char** argv)
{
	std::vector< std::string > file_list = getImagesList("../resources/images");
	ghog::lib::IHog* hog_cpu = new ghog::lib::HogCPU("hog.xml");
	ghog::lib::IHog* hog_gpu = new ghog::lib::HogGPU("hog.xml");
	cv::Size img_size(64, 64);
	cv::Size window_size(16, 16);
	int num_experiments = 10;

	boost::random::mt19937 random_gen;

	measure_time(hog_cpu, "Hog_CPU", file_list, img_size, window_size,
		num_experiments, random_gen);
	measure_time(hog_gpu, "Hog_GPU", file_list, img_size, window_size,
		num_experiments, random_gen);

	return 0;
}

