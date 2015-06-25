#include <iostream>
#include <vector>
#include <string>

#include <boost/chrono.hpp>
#include <boost/thread.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_smallint.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <include/HogDescriptor.h>
#include <include/HogGPU.h>

#include "utils.h"

std::vector< double > calculate_statistics(std::vector< double > values)
{
	std::vector< double > ret;
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
	average = total / ((float)values.size());
	for(int i = 0; i < values.size(); ++i)
	{
		std_dev += (values[i] - average) * (values[i] - average);
	}
	std_dev /= ((float)(values.size() - 1));
	std_dev = sqrt(std_dev);

	ret.push_back(average);
	ret.push_back(std_dev);
	ret.push_back(max);
	ret.push_back(min);
	return ret;
}

void report_statistics(std::vector< double > values,
	int ratio,
	std::string unit)
{
	std::vector< double > results = calculate_statistics(values);

	std::cout << "Average: " << results[0] * ratio << " " << unit << std::endl;
	std::cout << "Standard deviation: " << results[1] * ratio << std::endl;
	std::cout << "Max: " << results[2] * ratio << " " << unit << std::endl;
	std::cout << "Min: " << results[3] * ratio << " " << unit << std::endl;
	std::cout << std::endl;
}

void measure_time(ghog::lib::HogDescriptor* hog,
	std::string hog_name,
	std::vector< std::string > image_list,
	cv::Size img_size,
	cv::Size window_size,
	int num_experiments,
	boost::random::mt19937 random_gen)
{
	std::cout << "Running timing experiment on descriptor " << hog_name
		<< ", with " << image_list.size() << " images, using "
		<< num_experiments << " windows of size " << window_size << std::endl;

	cv::Mat input_img;
	cv::Mat grad_mag;
	cv::Mat grad_phase;
	cv::Mat descriptor;
	cv::Size descriptor_size(hog->get_descriptor_size(), 1);

	hog->alloc_buffer(img_size, CV_32FC3, input_img);
	hog->alloc_buffer(window_size, CV_32FC1, grad_mag);
	hog->alloc_buffer(window_size, CV_32FC1, grad_phase);
	hog->alloc_buffer(descriptor_size, CV_32FC1, descriptor);

	input_img.addref();
	grad_mag.addref();
	grad_phase.addref();
	descriptor.addref();

	boost::random::uniform_smallint< int > dist_w(1,
		input_img.cols - window_size.width - 2);
	boost::random::uniform_smallint< int > dist_h(1,
		input_img.rows - window_size.height - 2);

	boost::chrono::steady_clock::time_point start;
	boost::chrono::duration< double > time_elapsed;
	double time_elapsed_normalization;
	double time_elapsed_gradient;
	double time_elapsed_descriptor;

	std::vector< double > times_normalization;
	std::vector< double > times_gradient;
	std::vector< double > times_descriptor;
	std::vector< double > times_total;

	//Do some work, ignore the results, in order to discard GPU initialization time.
	cv::imread(image_list[0], CV_LOAD_IMAGE_COLOR).convertTo(input_img,
		CV_32FC3);
	input_img /= 256.0;

	for(int i = 0; i < 10; ++i)
	{
		int pos_x = dist_w(random_gen);
		int pos_y = dist_h(random_gen);

		hog->image_normalization_sync(input_img);
		hog->calc_gradient_sync(
			input_img.rowRange(pos_y, pos_y + window_size.height + 1).colRange(
				pos_x, pos_x + window_size.width + 1), grad_mag, grad_phase);
		hog->create_descriptor_sync(grad_mag, grad_phase, descriptor);
	}

	for(int i = 0; i < image_list.size(); ++i)
	{
		cv::imread(image_list[i], CV_LOAD_IMAGE_COLOR).convertTo(input_img,
			CV_32FC3);
		input_img /= 256.0;

		for(int j = 0; j < num_experiments; ++j)
		{
			int pos_x = dist_w(random_gen);
			int pos_y = dist_h(random_gen);

			start = boost::chrono::steady_clock::now();
			hog->image_normalization_sync(input_img);
			time_elapsed = boost::chrono::steady_clock::now() - start;
			time_elapsed_normalization = time_elapsed.count();
			hog->calc_gradient_sync(
				input_img.rowRange(pos_y, pos_y + window_size.height + 1)
					.colRange(pos_x, pos_x + window_size.width + 1), grad_mag,
				grad_phase);
			time_elapsed = boost::chrono::steady_clock::now() - start;
			time_elapsed_gradient = time_elapsed.count();
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
}

int count = 0;

double compare_matrices(cv::Mat m1,
	cv::Mat m2)
{
	double dot_product = 0.0;
	double mod_1 = 0.0;
	double mod_2 = 0.0;
//	double sum_squares_ref = 0.0;
//	double sum_squares_diff = 0.0;
//	cv::Mat diff = m1 - m2;
	for(int i = 0; i < m1.rows; ++i)
	{
		for(int j = 0; j < m1.cols; ++j)
		{
			double m1_ij = (double)m1.at< float >(i, j);
			double m2_ij = (double)m1.at< float >(i, j);
			mod_1 += m1_ij * m1_ij;
			mod_2 += m2_ij * m2_ij;
			dot_product += m1_ij * m2_ij;
//			double diff_ij = (double)diff.at< float >(i, j);
//			sum_squares_ref += m1_ij * m1_ij;
//			sum_squares_diff += diff_ij * diff_ij;
		}
	}
	if(count == 50)
	{
		std::cout << count << ": " << "Mod 1: " << mod_1 << "  Mod 2: " << mod_2
			<< "  dot: " << dot_product << std::endl;
	}
	count++;
	mod_1 = sqrt(mod_1);
	mod_2 = sqrt(mod_2);
	return dot_product / (mod_1 * mod_2);
//	sum_squares_diff /= sum_squares_ref;
//	return 100 * sqrt(sum_squares_diff);
}

void measure_difference(ghog::lib::HogDescriptor* hog1,
	ghog::lib::HogDescriptor* hog2,
	std::string hog_name1,
	std::string hog_name2,
	std::vector< std::string > image_list,
	cv::Size img_size,
	cv::Size window_size,
	int num_experiments,
	boost::random::mt19937 random_gen)
{
	std::cout << "Running difference experiment on descriptors " << hog_name1
		<< " and " << hog_name2 << ", with " << image_list.size()
		<< " images, using " << num_experiments << " windows of size "
		<< window_size << std::endl;

	cv::Size descriptor_size(hog1->get_descriptor_size(), 1);

	cv::Mat input_img1;
	cv::Mat grad_mag1;
	cv::Mat grad_phase1;
	cv::Mat descriptor1;

	hog1->alloc_buffer(img_size, CV_32FC3, input_img1);
	hog1->alloc_buffer(window_size, CV_32FC1, grad_mag1);
	hog1->alloc_buffer(window_size, CV_32FC1, grad_phase1);
	hog1->alloc_buffer(descriptor_size, CV_32FC1, descriptor1);

	input_img1.addref();
	grad_mag1.addref();
	grad_phase1.addref();
	descriptor1.addref();

	cv::Mat input_img2;
	cv::Mat grad_mag2;
	cv::Mat grad_phase2;
	cv::Mat descriptor2;

	hog2->alloc_buffer(img_size, CV_32FC3, input_img2);
	hog2->alloc_buffer(window_size, CV_32FC1, grad_mag2);
	hog2->alloc_buffer(window_size, CV_32FC1, grad_phase2);
	hog2->alloc_buffer(descriptor_size, CV_32FC1, descriptor2);

	input_img2.addref();
	grad_mag2.addref();
	grad_phase2.addref();
	descriptor2.addref();

	boost::random::uniform_smallint< int > dist_w(1,
		input_img1.cols - window_size.width - 2);
	boost::random::uniform_smallint< int > dist_h(1,
		input_img1.rows - window_size.height - 2);

//	std::vector< double > errors_normalization;
	std::vector< double > magnitude_similarity;
	std::vector< double > phase_similarity;
	std::vector< double > descriptor_similarity;
	std::vector< double > total_similarity;

	std::cout << "Calculating partial difference" << std::endl;

	for(int i = 0; i < image_list.size(); ++i)
	{
		cv::imread(image_list[i], CV_LOAD_IMAGE_COLOR).convertTo(input_img1,
			CV_32FC3);
		cv::imread(image_list[i], CV_LOAD_IMAGE_COLOR).convertTo(input_img2,
			CV_32FC3);
		input_img1 /= 256.0;
		input_img2 /= 256.0;

		for(int j = 0; j < num_experiments; ++j)
		{
			int pos_x = dist_w(random_gen);
			int pos_y = dist_h(random_gen);

//			hog1->image_normalization_sync(input_img1);
//			hog2->image_normalization_sync(input_img2);
//			errors_normalization.push_back(
//				compare_matrices(input_img1, input_img2));

			hog1->calc_gradient_sync(
				input_img1.rowRange(pos_y, pos_y + window_size.height).colRange(
					pos_x, pos_x + window_size.width), grad_mag1, grad_phase1);
			hog2->calc_gradient_sync(
				input_img2.rowRange(pos_y, pos_y + window_size.height).colRange(
					pos_x, pos_x + window_size.width), grad_mag2, grad_phase2);
			magnitude_similarity.push_back(
				compare_matrices(grad_mag1, grad_mag2));
			phase_similarity.push_back(
				compare_matrices(grad_phase1, grad_phase2));

			grad_mag1.copyTo(grad_mag2);
			grad_phase1.copyTo(grad_phase2);

			hog1->create_descriptor_sync(grad_mag1, grad_phase1, descriptor1);
			hog2->create_descriptor_sync(grad_mag2, grad_phase2, descriptor2);

			if(count == 50)
			{
//				cv::normalize(grad_mag1, grad_mag1, 0.0, 1.0, cv::NORM_MINMAX,
//					CV_32FC1);
//				cv::normalize(grad_mag2, grad_mag2, 0.0, 1.0, cv::NORM_MINMAX,
//					CV_32FC1);
//				cv::normalize(grad_phase1, grad_phase1, 0.0, 1.0,
//					cv::NORM_MINMAX, CV_32FC1);
//				cv::normalize(grad_phase2, grad_phase2, 0.0, 1.0,
//					cv::NORM_MINMAX, CV_32FC1);
//
//				cv::imshow("i1",
//					input_img1.rowRange(pos_y, pos_y + window_size.height)
//						.colRange(pos_x, pos_x + window_size.width));
//				cv::imshow("i2",
//					input_img2.rowRange(pos_y, pos_y + window_size.height)
//						.colRange(pos_x, pos_x + window_size.width));
//				cv::imshow("m1", grad_mag1);
//				cv::imshow("m2", grad_mag2);
//				cv::imshow("p1", grad_phase1);
//				cv::imshow("p2", grad_phase2);

				cv::Mat input_1_tmp = input_img1.rowRange(pos_y,
					pos_y + window_size.height).colRange(pos_x,
					pos_x + window_size.width);

				cv::Mat input_2_tmp = input_img2.rowRange(pos_y,
					pos_y + window_size.height).colRange(pos_x,
					pos_x + window_size.width);

				for(int i = 0; i < input_1_tmp.rows; ++i)
				{
					for(int j = 0; j < input_1_tmp.cols; ++j)
					{
						if((i >= 80) && (i <= 87) && (j >= 40) && (j <= 47))
						{
							std::cout << i << " " << j << "->  Image 1: "
								<< input_1_tmp.at< float >(i, j) << "  Mag 1: "
								<< grad_mag1.at< float >(i, j) << "  Phase 1: "
								<< grad_phase1.at< float >(i, j) << std::endl;
							std::cout << i << " " << j << "->  Image 2: "
								<< input_2_tmp.at< float >(i, j) << "  Mag 2: "
								<< grad_mag2.at< float >(i, j) << "  Phase 2: "
								<< grad_phase2.at< float >(i, j) << std::endl;
						}
					}
				}

				std::cout << std::endl << std::endl;
			for(int i = 0; i < descriptor1.cols; ++i)
			{
				std::cout << i << "-> desc1: " << descriptor1.at< float >(i)
					<< "  desc2: " << descriptor2.at< float >(i) << std::endl;
			}
			std::cout << std::endl << std::endl;

//				cv::waitKey();
			}

			descriptor_similarity.push_back(
				compare_matrices(descriptor1, descriptor2));
		}
	}

//	std::cout << "Error on image normalization:" << std::endl;
//	report_statistics(errors_normalization, 1, "\%");
	std::cout << "Error on magnitude calculation:" << std::endl;
	report_statistics(magnitude_similarity, 1, "similarity");
	std::cout << "Error on phase calculation:" << std::endl;
	report_statistics(phase_similarity, 1, "similarity");
	std::cout << "Error on descriptor calculation:" << std::endl;
	report_statistics(descriptor_similarity, 1, "similarity");

//	std::cout << "Magnitude similarity:" << std::endl;
//	for(int i = 0; i < magnitude_similarity.size(); ++i)
//	{
//		std::cout << magnitude_similarity[i] << "  ";
//	}
//	std::cout << std::endl;
//	std::cout << "Phase similarity:" << std::endl;
//	for(int i = 0; i < phase_similarity.size(); ++i)
//	{
//		std::cout << phase_similarity[i] << "  ";
//	}
//	std::cout << std::endl;

	std::cout << "Calculating complete difference" << std::endl;

	for(int i = 0; i < image_list.size(); ++i)
	{
		cv::imread(image_list[i], CV_LOAD_IMAGE_COLOR).convertTo(input_img1,
			CV_32FC3);
		cv::imread(image_list[i], CV_LOAD_IMAGE_COLOR).convertTo(input_img2,
			CV_32FC3);
		input_img1 /= 256.0;
		input_img2 /= 256.0;

		for(int j = 0; j < num_experiments; ++j)
		{
			int pos_x = dist_w(random_gen);
			int pos_y = dist_h(random_gen);

//			hog1->image_normalization_sync(input_img1);
			hog1->calc_gradient_sync(
				input_img1.rowRange(pos_y, pos_y + window_size.height).colRange(
					pos_x, pos_x + window_size.width), grad_mag1, grad_phase1);
			hog1->create_descriptor_sync(grad_mag1, grad_phase1, descriptor1);

//			hog2->image_normalization_sync(input_img2);
			hog2->calc_gradient_sync(
				input_img2.rowRange(pos_y, pos_y + window_size.height).colRange(
					pos_x, pos_x + window_size.width), grad_mag2, grad_phase2);
			hog2->create_descriptor_sync(grad_mag2, grad_phase2, descriptor2);

			total_similarity.push_back(
				compare_matrices(descriptor1, descriptor2));
		}
	}

	std::cout << "Total error :" << std::endl;
	report_statistics(total_similarity, 1, "similarity");
	std::cout << std::endl;

	std::cout << "Descriptor similarity:" << std::endl;
	for(int i = 0; i < descriptor_similarity.size(); ++i)
	{
		std::cout << descriptor_similarity[i] << "  ";
	}
	std::cout << std::endl;

	std::cout << "Total similarity:" << std::endl;
	for(int i = 0; i < total_similarity.size(); ++i)
	{
		std::cout << total_similarity[i] << "  ";
	}
	std::cout << std::endl;
}

int main(int argc,
	char** argv)
{
	setenv("DISPLAY", ":0", 0);
	std::vector< std::string > file_list = getImagesList("../resources/images");
	ghog::lib::HogDescriptor* hog_cpu = new ghog::lib::HogDescriptor("hog.xml");
	ghog::lib::HogDescriptor* hog_gpu = new ghog::lib::HogGPU("hog.xml");
	cv::Size img_size(3648, 2736);
	cv::Size window_size(64, 128);
	int num_experiments = 10;

	boost::random::mt19937 random_gen;

//	measure_time(hog_cpu, "Hog_CPU", file_list, img_size, window_size,
//		num_experiments, random_gen);
//	measure_time(hog_gpu, "Hog_GPU", file_list, img_size, window_size,
//		num_experiments, random_gen);
	measure_difference(hog_cpu, hog_gpu, "Hog CPU", "Hog GPU", file_list,
		img_size, window_size, num_experiments, random_gen);

	return 0;
}

