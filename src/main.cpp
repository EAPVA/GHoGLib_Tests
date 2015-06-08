#include <iostream>

#include <boost/chrono.hpp>
#include <boost/thread.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <include/HogCPU.h>
#include <include/HogGPU.h>

#include "utils.h"

class ResizeCallback: public ghog::lib::ImageCallback
{
public:
	ResizeCallback(int num_images)
	{
		_finished = 0;
		_total_images = num_images;
	}

	void image_processed(cv::Mat original,
		cv::Mat processed)
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
	std::vector< std::string > file_list = getImagesList("resources/images");
	ResizeCallback callback(file_list.size());
	ghog::lib::IHog* hog_cpu = new ghog::lib::HogCPU("hog.xml");
	ghog::lib::IHog* hog_gpu = new ghog::lib::HogGPU("hog.xml");
	cv::Size new_size(24, 24);

	boost::chrono::steady_clock::time_point start;
	boost::chrono::duration< float > time_elapsed;

	std::cout << "Loading and processing images on the CPU." << std::endl;
	start = boost::chrono::steady_clock::now();
	for(int i = 0; i < file_list.size(); ++i)
	{
		cv::Mat input_img = cv::imread(file_list[i], CV_LOAD_IMAGE_GRAYSCALE);
		hog_cpu->resize(input_img, new_size, &callback);
	}
	std::cout << "Processing " << file_list.size() << " images..." << std::endl;
	while(!callback.is_finished())
	{
		std::cout << ".";
		std::cout.flush();
		boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
	}
	time_elapsed = boost::chrono::steady_clock::now() - start;
	std::cout << std::endl;
	std::cout << "Images processed in " << time_elapsed << std::endl;
	std::cout << std::endl << std::endl;

	callback.reset();

	std::cout << "Loading and processing images on the GPU." << std::endl;
	start = boost::chrono::steady_clock::now();
	for(int i = 0; i < file_list.size(); ++i)
	{
		cv::Mat input_img = cv::imread(file_list[i], CV_LOAD_IMAGE_GRAYSCALE);
		hog_gpu->resize(input_img, new_size, &callback);
	}
	std::cout << "Processing " << file_list.size() << " images..." << std::endl;
	while(!callback.is_finished())
	{
		std::cout << ".";
		std::cout.flush();
		boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
	}
	time_elapsed = boost::chrono::steady_clock::now() - start;
	std::cout << std::endl;
	std::cout << "Images processed in " << time_elapsed << std::endl;
	return 0;
}

