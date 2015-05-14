#include <iostream>

#include <boost/thread.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <include/ImageUtils.h>

#include "utils.h"

class ResizeCallback: public ghog::lib::ImageCallback
{
public:
	ResizeCallback(int num_images)
	{
		_finished = 0;
		_total_images = num_images;
	}
	void image_processed(cv::Mat ret_mat)
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
	std::vector< std::string > file_list = getImagesList("resources/inputs");
	cleanOutputDir("resources/resized");
	ResizeCallback callback(file_list.size());
	ghog::lib::ImageUtils utils(&callback);
	cv::Size new_size(24, 24);

	for(int i = 0; i < file_list.size(); ++i)
	{
		cv::Mat input_img = cv::imread(file_list[i], CV_LOAD_IMAGE_GRAYSCALE);
		utils.resize(input_img, new_size);
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

