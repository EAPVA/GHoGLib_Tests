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

class MyCallback: public ghog::lib::GradientCallback
{
public:
	MyCallback(int num_images)
	{
		_finished = 0;
		_total_images = num_images;
	}

	void gradients_obtained(cv::Mat original,
		cv::Mat gradients_magnitude,
		cv::Mat gradients_phase)
	{
		_finished++;
	}

	bool is_finished()
	{
		return (_finished == _total_images);
	}

	void reset()
	{
		_finished = 0;
	}

private:
	int _finished;
	int _total_images;
};

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

	MyCallback callback(1);
	cv::Mat input_img = hog->alloc_buffer(img_size, CV_32FC1, 0);
	cv::Mat grad_mag = hog->alloc_buffer(window_size, CV_32FC1, 0);
	cv::Mat grad_phase = hog->alloc_buffer(window_size, CV_32FC1, 0);
	cv::Mat input_window;

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
			input_window = input_img.rowRange(pos_y,
				pos_y + window_size.height + 1).colRange(pos_x,
				pos_x + window_size.width + 1);

			start = boost::chrono::steady_clock::now();
			hog->calc_gradient(input_window, grad_mag, grad_phase, &callback);
			while(!callback.is_finished())
			{
				boost::this_thread::yield();
			}
			time_elapsed = boost::chrono::steady_clock::now() - start;
			callback.reset();
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
	cv::Size img_size(3648, 2736);
	cv::Size window_size(100, 100);
	int num_experiments = 1000;

	boost::random::mt19937 random_gen;

	measure_time(hog_cpu, "Hog_CPU", file_list, img_size, window_size,
		num_experiments, random_gen);
	measure_time(hog_gpu, "Hog_GPU", file_list, img_size, window_size,
		num_experiments, random_gen);

	return 0;
}

