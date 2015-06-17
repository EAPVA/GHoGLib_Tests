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

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

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

	std::cout << "Before first imread:" << std::endl;
	std::cout << "Input image:" << std::endl;
	std::cout << "Step: " << input_img.step1() << "  " << "Addr: "
		<< ((long long)input_img.data) << std::endl;
	std::cout << "Magnitude matrix:" << std::endl;
	std::cout << "Step: " << grad_mag.step1() << "  " << "Addr: "
		<< ((long long)grad_mag.data) << std::endl;
	std::cout << "Phase matrix:" << std::endl;
	std::cout << "Step: " << grad_phase.step1() << "  " << "Addr: "
		<< ((long long)grad_phase.data) << std::endl;
	std::cout << std::endl << std::endl;

	boost::chrono::steady_clock::time_point start;
	boost::chrono::duration< double > time_elapsed;

	boost::random::uniform_smallint< int > dist_w(1,
		input_img.cols - window_size.width - 2);
	boost::random::uniform_smallint< int > dist_h(1,
		input_img.rows - window_size.height - 2);

	std::vector< double > values;

	for(int i = 0; i < image_list.size(); ++i)
	{
		std::cout << "read image" << std::endl;
		cv::imread(image_list[i], CV_LOAD_IMAGE_GRAYSCALE).convertTo(input_img,
			CV_32FC1);

		std::cout << "display image" << std::endl;

		for(int j = 0; j < input_img.rows; ++j)
		{
			std::cout << j << ": ";
			std::cout.flush();
			for(int k = 0; k < input_img.cols; ++k)
			{
				std::cout << k << " ";
				std::cout << input_img.at< float >(j, k) << "  ";
				std::cout.flush();
			}
			std::cout << std::endl;
		}

		std::cout << "imread #" << i << std::endl;
		std::cout << "Input image:" << std::endl;
		std::cout << "Step: " << input_img.step1() << "  " << "Addr: "
			<< ((long long)input_img.data) << std::endl;
		std::cout << "Magnitude matrix:" << std::endl;
		std::cout << "Step: " << grad_mag.step1() << "  " << "Addr: "
			<< ((long long)grad_mag.data) << std::endl;
		std::cout << "Phase matrix:" << std::endl;
		std::cout << "Step: " << grad_phase.step1() << "  " << "Addr: "
			<< ((long long)grad_phase.data) << std::endl;
		std::cout << std::endl << std::endl;

		for(int j = 0; j < num_experiments; ++j)
		{
			std::cout << "Experiment #" << j << std::endl;
			std::cout << "Input image:" << std::endl;
			std::cout << "Step: " << input_img.step1() << "  " << "Addr: "
				<< ((long long)input_img.data) << std::endl;
			std::cout << "Magnitude matrix:" << std::endl;
			std::cout << "Step: " << grad_mag.step1() << "  " << "Addr: "
				<< ((long long)grad_mag.data) << std::endl;
			std::cout << "Phase matrix:" << std::endl;
			std::cout << "Step: " << grad_phase.step1() << "  " << "Addr: "
				<< ((long long)grad_phase.data) << std::endl;
			std::cout << std::endl << std::endl;

			int pos_x = dist_w(random_gen);
			int pos_y = dist_h(random_gen);

			start = boost::chrono::steady_clock::now();
//			hog->calc_gradient(
//				input_img.rowRange(pos_y, pos_y + window_size.height + 1)
//					.colRange(pos_x, pos_x + window_size.width + 1), grad_mag,
//				grad_phase, &callback);
//			while(!callback.is_finished())
//			{
//				boost::this_thread::yield();
//			}
			time_elapsed = boost::chrono::steady_clock::now() - start;
			callback.reset();
			values.push_back(time_elapsed.count());
		}
	}

	report_statistics(values, "seconds");
}

void teste()
{
	std::cout << "Allocating buffer" << std::endl;
	cv::gpu::CudaMem cmem(64, 64, CV_32FC1, cv::gpu::CudaMem::ALLOC_ZEROCOPY);
	std::cout << "Creating header" << std::endl;
	cv::Mat mat = cmem.createMatHeader();

	std::cout << "Before imread:" << std::endl;
	std::cout << "Input image:" << std::endl;
	std::cout << "Step: " << mat.step1() << "  " << "Addr: "
		<< ((long long)mat.data) << std::endl;
	std::cout << "Loading image" << std::endl;
	cv::imread("teste.png", CV_LOAD_IMAGE_GRAYSCALE).convertTo(mat, CV_32FC1);

	std::cout << "After imread:" << std::endl;
	std::cout << "Input image:" << std::endl;
	std::cout << "Step: " << mat.step1() << "  " << "Addr: "
		<< ((long long)mat.data) << std::endl;

	std::cout << "Finish" << std::endl;
}

int main(int argc,
	char** argv)
{
	teste();
	return 0;

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

