#include "difference.h"

#include <boost/random/uniform_smallint.hpp>
#include <boost/lexical_cast.hpp>

#include <opencv2/highgui/highgui.hpp>

#include "stats.h"

double compare_matrices(cv::Mat m1,
	cv::Mat m2)
{
	double distance = 0.0;
	double magnitude_1 = 0.0;
	double magnitude_2 = 0.0;
	int num_different = 0;
	for(int i = 0; i < m1.rows; ++i)
	{
		for(int j = 0; j < m1.cols; ++j)
		{
			double m1_ij = (double)m1.at< float >(i, j);
			double m2_ij = (double)m2.at< float >(i, j);
			magnitude_1 += (m1_ij * m1_ij);
			magnitude_2 += (m2_ij * m2_ij);
			double difference = abs(m1_ij - m2_ij);
			distance += difference * difference;
		}
	}
	double sum_magnitudes = sqrt(magnitude_1) + sqrt(magnitude_2);
	return sqrt(distance) / sum_magnitudes;
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
	cv::Mat normalized_img1;
	cv::Mat grad_mag1;
	cv::Mat grad_phase1;
	cv::Mat descriptor1;

	hog1->alloc_buffer(img_size, CV_32FC3, input_img1);
	hog1->alloc_buffer(window_size, CV_32FC3, normalized_img1);
	hog1->alloc_buffer(window_size, CV_32FC1, grad_mag1);
	hog1->alloc_buffer(window_size, CV_32FC1, grad_phase1);
	hog1->alloc_buffer(descriptor_size, CV_32FC1, descriptor1);

	cv::Mat input_img2;
	cv::Mat normalized_img2;
	cv::Mat grad_mag2;
	cv::Mat grad_phase2;
	cv::Mat descriptor2;

	hog2->alloc_buffer(img_size, CV_32FC3, input_img2);
	hog2->alloc_buffer(window_size, CV_32FC3, normalized_img2);
	hog2->alloc_buffer(window_size, CV_32FC1, grad_mag2);
	hog2->alloc_buffer(window_size, CV_32FC1, grad_phase2);
	hog2->alloc_buffer(descriptor_size, CV_32FC1, descriptor2);

	boost::random::uniform_smallint< int > dist_w(1,
		input_img1.cols - window_size.width - 2);
	boost::random::uniform_smallint< int > dist_h(1,
		input_img1.rows - window_size.height - 2);

	std::vector< double > errors_normalization;
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

			input_img1.rowRange(pos_y, pos_y + window_size.height).colRange(
				pos_x, pos_x + window_size.width).copyTo(normalized_img1);
			input_img2.rowRange(pos_y, pos_y + window_size.height).colRange(
				pos_x, pos_x + window_size.width).copyTo(normalized_img2);

			hog1->image_normalization_sync(normalized_img1);
			hog2->image_normalization_sync(normalized_img2);
			errors_normalization.push_back(
				compare_matrices(normalized_img1, normalized_img2));

			normalized_img1.copyTo(normalized_img2);

			hog1->calc_gradient_sync(normalized_img1, grad_mag1, grad_phase1);
			hog2->calc_gradient_sync(normalized_img2, grad_mag2, grad_phase2);
			magnitude_similarity.push_back(
				compare_matrices(grad_mag1, grad_mag2));
			phase_similarity.push_back(
				compare_matrices(grad_phase1, grad_phase2));

			grad_mag1.copyTo(grad_mag2);
			grad_phase1.copyTo(grad_phase2);

			hog1->create_descriptor_sync(grad_mag1, grad_phase1, descriptor1);
			hog2->create_descriptor_sync(grad_mag2, grad_phase2, descriptor2);

			descriptor_similarity.push_back(
				compare_matrices(descriptor1, descriptor2));
		}
	}

	std::cout << "Error on image normalization:" << std::endl;
	report_statistics(errors_normalization, 1, "normalized euclidian distance");
	std::cout << "Error on magnitude calculation:" << std::endl;
	report_statistics(magnitude_similarity, 1, "normalized euclidian distance");
	std::cout << "Error on phase calculation:" << std::endl;
	report_statistics(phase_similarity, 1, "normalized euclidian distance");
	std::cout << "Error on descriptor calculation:" << std::endl;
	report_statistics(descriptor_similarity, 1,
		"normalized euclidian distance");

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

			input_img1.rowRange(pos_y, pos_y + window_size.height).colRange(
				pos_x, pos_x + window_size.width).copyTo(normalized_img1);
			input_img2.rowRange(pos_y, pos_y + window_size.height).colRange(
				pos_x, pos_x + window_size.width).copyTo(normalized_img2);

			hog1->image_normalization_sync(input_img1);
			hog1->calc_gradient_sync(normalized_img1, grad_mag1, grad_phase1);
			hog1->create_descriptor_sync(grad_mag1, grad_phase1, descriptor1);

			hog2->image_normalization_sync(input_img2);
			hog2->calc_gradient_sync(normalized_img2, grad_mag2, grad_phase2);
			hog2->create_descriptor_sync(grad_mag2, grad_phase2, descriptor2);

			total_similarity.push_back(
				compare_matrices(descriptor1, descriptor2));
		}
	}

	std::cout << "Total similarity :" << std::endl;
	report_statistics(total_similarity, 1, "normalized euclidian distance");
	std::cout << std::endl;
}
