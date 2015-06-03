#include <iostream>

#include <boost/thread.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <include/HogCPU.h>

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

private:
	int _finished;
	int _total_images;
};

int main(int argc,
	char** argv)
{
	std::vector<std::string> file_list = getImagesList("resources/images");
	ResizeCallback callback(file_list.size());
	ghog::lib::IHog* utils = new ghog::lib::HogCPU("hog.xml");
	cv::Size new_size(24, 24);

	for(int i = 0; i < file_list.size(); ++i)
	{
		cv::Mat input_img = cv::imread(file_list[i], CV_LOAD_IMAGE_GRAYSCALE);
		utils->resize(input_img, new_size, &callback);
	}
	std::cout << "Processing " << file_list.size() << " images..." << std::endl;
	while(!callback.is_finished())
	{
		std::cout << ".";
		std::cout.flush();
		boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
	}
	std::cout << std::endl;
	std::cout << "Images processed." << std::endl;
	return 0;
}

