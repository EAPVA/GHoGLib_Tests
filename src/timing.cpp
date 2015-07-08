#include "timing.h"

#include <iostream>

#include <boost/chrono.hpp>
#include <boost/random/uniform_smallint.hpp>
#include <boost/lexical_cast.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "stats.h"

double measure_time(ghog::lib::HogDescriptor* hog,
	std::string hog_name,
	std::vector< std::string > image_list,
	cv::Size img_size,
	int num_experiments,
	boost::random::mt19937 random_gen)
{
	cv::Mat input_img;
	cv::Mat normalized_img;
	cv::Mat grad_mag;
	cv::Mat grad_phase;
	cv::Mat descriptor;
	cv::Size descriptor_size(hog->get_descriptor_size(), 1);
	cv::Size window_size;

	hog->alloc_buffer(img_size, CV_32FC3, input_img);

	boost::chrono::steady_clock::time_point start;
	boost::chrono::duration< double > time_elapsed;
	double time_elapsed_normalization;
	double time_elapsed_gradient;
	double time_elapsed_descriptor;

	std::vector< double > times_normalization;
	std::vector< double > times_gradient;
	std::vector< double > times_descriptor;
	std::vector< double > times_total;

	cv::imread(image_list[0], CV_LOAD_IMAGE_COLOR).convertTo(input_img,
		CV_32FC3);
	input_img /= 256.0;

	int i = 15;

	window_size.width = 8 * 16 * i;
	window_size.height = 8 * 9 * i;

	boost::random::uniform_smallint< int > dist_w(1,
		input_img.cols - window_size.width - 2);
	boost::random::uniform_smallint< int > dist_h(1,
		input_img.rows - window_size.height - 2);

	std::cout << "Running timing experiment#" << i << " on descriptor "
		<< hog_name << ", using " << num_experiments << " windows of size "
		<< window_size << std::endl;

	descriptor_size.width = hog->get_descriptor_size();
	hog->alloc_buffer(window_size, CV_32FC3, normalized_img);
	hog->alloc_buffer(window_size, CV_32FC1, grad_mag);
	hog->alloc_buffer(window_size, CV_32FC1, grad_phase);
	hog->alloc_buffer(descriptor_size, CV_32FC1, descriptor);

	for(int j = 0; j < 10; ++j)
	{
		int pos_x = dist_w(random_gen);
		int pos_y = dist_h(random_gen);

		input_img.rowRange(pos_y, pos_y + window_size.height).colRange(pos_x,
			pos_x + window_size.width).copyTo(normalized_img);

		hog->image_normalization_sync(normalized_img);
		hog->calc_gradient_sync(normalized_img, grad_mag, grad_phase);
		hog->create_descriptor_sync(grad_mag, grad_phase, descriptor);
	}

	for(int j = 0; j < num_experiments; ++j)
	{
		int pos_x = dist_w(random_gen);
		int pos_y = dist_h(random_gen);

		input_img.rowRange(pos_y, pos_y + window_size.height).colRange(pos_x,
			pos_x + window_size.width).copyTo(normalized_img);

		start = boost::chrono::steady_clock::now();
		hog->image_normalization_sync(normalized_img);
		time_elapsed = boost::chrono::steady_clock::now() - start;
		time_elapsed_normalization = time_elapsed.count();

		start = boost::chrono::steady_clock::now();
		hog->calc_gradient_sync(normalized_img, grad_mag, grad_phase);
		time_elapsed = boost::chrono::steady_clock::now() - start;
		time_elapsed_gradient = time_elapsed.count();

		start = boost::chrono::steady_clock::now();
		hog->create_descriptor_sync(grad_mag, grad_phase, descriptor);
		time_elapsed = boost::chrono::steady_clock::now() - start;
		time_elapsed_descriptor = time_elapsed.count();

		times_normalization.push_back(time_elapsed_normalization);
		times_gradient.push_back(time_elapsed_gradient);
		times_descriptor.push_back(time_elapsed_descriptor);
		times_total.push_back(
			(time_elapsed_normalization + time_elapsed_gradient
				+ time_elapsed_descriptor));
	}

	std::cout << "Time spent on normalization:" << std::endl;
	report_statistics(times_normalization, 1000, "milliseconds");
	std::cout << "Time spent on gradient:" << std::endl;
	report_statistics(times_gradient, 1000, "milliseconds");
	std::cout << "Time spent on descriptor:" << std::endl;
	report_statistics(times_descriptor, 1000, "milliseconds");
	std::cout << "Time spent total:" << std::endl;
	report_statistics(times_total, 1000, "milliseconds");
	std::cout << std::endl;

	times_normalization.clear();
	times_gradient.clear();
	times_descriptor.clear();
	times_total.clear();

	return 0.0;
}

double measure_time_opencv(std::vector< std::string > image_list,
	cv::Size img_size,
	int num_experiments,
	boost::random::mt19937 random_gen)
{
	boost::chrono::steady_clock::time_point start;
	boost::chrono::duration< double > time_elapsed;

	std::vector< double > times_total;

	for(int i = 1; i < 16; ++i)
	{
		cv::Size window_size(8 * 16 * i, 8 * 9 * i);

		std::cout
			<< "Running timing experiment on opencv's CPU Hog Descriptor, with "
			<< image_list.size() << " images, using " << num_experiments
			<< " windows of size " << window_size << std::endl;

		cv::HOGDescriptor cpu_hog(window_size, cv::Size(16, 16), cv::Size(8, 8),
			cv::Size(8, 8), 9);
		cv::gpu::HOGDescriptor gpu_hog(window_size);

		cv::Mat input_img(img_size, CV_8UC4);
		cv::Mat test_image(window_size, CV_8UC4);
		cv::gpu::GpuMat gpu_test_image(window_size, CV_8UC4);
		std::vector< float > cpu_descriptor;
		cv::Mat descriptor;
		cv::gpu::GpuMat gpu_descriptor;

		boost::random::uniform_smallint< int > dist_w(1,
			img_size.width - window_size.width - 2);
		boost::random::uniform_smallint< int > dist_h(1,
			img_size.height - window_size.height - 2);

		for(int j = 0; j < image_list.size(); ++j)
		{
			cv::imread(image_list[j], CV_LOAD_IMAGE_COLOR).convertTo(input_img,
				CV_8UC4);

			for(int k = 0; k < num_experiments; ++k)
			{
				int pos_x = dist_w(random_gen);
				int pos_y = dist_h(random_gen);

				input_img.rowRange(pos_y, pos_y + window_size.height).colRange(
					pos_x, pos_x + window_size.width).copyTo(test_image);

				start = boost::chrono::steady_clock::now();
				cpu_hog.compute(test_image, cpu_descriptor, cv::Size(8, 8),
					cv::Size(0, 0));
				time_elapsed = boost::chrono::steady_clock::now() - start;
				times_total.push_back(time_elapsed.count());
			}
		}
		std::cout << "Time spent total:" << std::endl;
		report_statistics(times_total, 1000, "milliseconds");
		std::cout << std::endl;

		times_total.clear();

		std::cout
			<< "Running timing experiment on opencv's GPU Hog Descriptor, with "
			<< image_list.size() << " images, using " << num_experiments
			<< " windows of size " << window_size << std::endl;

		for(int j = 0; j < image_list.size(); ++j)
		{
			cv::imread(image_list[j], CV_LOAD_IMAGE_UNCHANGED).convertTo(
				input_img, CV_8UC4);

			for(int k = 0; k < 10; ++k)
			{
				int pos_x = dist_w(random_gen);
				int pos_y = dist_h(random_gen);

				input_img.rowRange(pos_y, pos_y + window_size.height).colRange(
					pos_x, pos_x + window_size.width).copyTo(test_image);

				gpu_test_image.upload(test_image);
				gpu_hog.getDescriptors(gpu_test_image, cv::Size(8, 8),
					gpu_descriptor);
				gpu_descriptor.download(descriptor);
			}

			for(int k = 0; k < num_experiments; ++k)
			{
				int pos_x = dist_w(random_gen);
				int pos_y = dist_h(random_gen);

				input_img.rowRange(pos_y, pos_y + window_size.height).colRange(
					pos_x, pos_x + window_size.width).copyTo(test_image);

				start = boost::chrono::steady_clock::now();
				gpu_test_image.upload(test_image);
				gpu_hog.getDescriptors(gpu_test_image, cv::Size(8, 8),
					gpu_descriptor);
				gpu_descriptor.download(descriptor);
				time_elapsed = boost::chrono::steady_clock::now() - start;
				times_total.push_back(time_elapsed.count());
			}
		}

		std::cout << "Time spent total:" << std::endl;
		report_statistics(times_total, 1000, "milliseconds");
		std::cout << std::endl;

		times_total.clear();
	}

	return 0.0;
}
