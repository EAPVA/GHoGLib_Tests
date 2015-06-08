#include <iostream>

#include <boost/chrono.hpp>
#include <boost/thread.hpp>

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

int main(int argc,
	char** argv)
{
	std::vector< std::string > file_list = getImagesList("../resources/images");
	MyCallback callback(1);
	ghog::lib::IHog* hog_cpu = new ghog::lib::HogCPU("hog.xml");
	ghog::lib::IHog* hog_gpu = new ghog::lib::HogGPU("hog.xml");

	cv::Mat grad_mag(3648, 2736, CV_32FC1);
	cv::Mat grad_phase(3648, 2736, CV_32FC1);

	boost::chrono::steady_clock::time_point start;
	boost::chrono::duration< float > time_elapsed;

	std::cout << "Processing " << file_list.size() << " images on the CPU"
		<< std::endl;
	start = boost::chrono::steady_clock::now();
	for(int i = 0; i < file_list.size(); ++i)
	{
		cv::Mat input_img = cv::imread(file_list[i], CV_LOAD_IMAGE_GRAYSCALE);
		input_img.convertTo(input_img, CV_32FC1);
		hog_cpu->calc_gradient(input_img, grad_mag, grad_phase, &callback);
		while(!callback.is_finished())
		{
			boost::this_thread::sleep(boost::posix_time::milliseconds(100));
		}
		callback.reset();
	}

	time_elapsed = boost::chrono::steady_clock::now() - start;
	std::cout << std::endl;
	std::cout << "Images processed in " << time_elapsed << std::endl;
	std::cout << std::endl << std::endl;

	callback.reset();

	std::cout << "Processing " << file_list.size() << " images on the GPU"
		<< std::endl;
	start = boost::chrono::steady_clock::now();
	for(int i = 0; i < file_list.size(); ++i)
	{
		cv::Mat input_img = cv::imread(file_list[i], CV_LOAD_IMAGE_GRAYSCALE);
		input_img.convertTo(input_img, CV_32FC1);
		hog_gpu->calc_gradient(input_img, grad_mag, grad_phase, &callback);
		while(!callback.is_finished())
		{
			boost::this_thread::sleep(boost::posix_time::milliseconds(100));
		}
		callback.reset();
	}

	time_elapsed = boost::chrono::steady_clock::now() - start;
	std::cout << std::endl;
	std::cout << "Images processed in " << time_elapsed << std::endl;

	return 0;
}

