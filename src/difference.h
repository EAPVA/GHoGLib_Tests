/*
 * difference.h
 *
 *  Created on: Jun 29, 2015
 *      Author: teider
 */

#ifndef DIFFERENCE_H_
#define DIFFERENCE_H_

#include <string>

#include <boost/random/mersenne_twister.hpp>

#include <opencv2/core/core.hpp>

#include <include/HogDescriptor.h>

void measure_difference(ghog::lib::HogDescriptor* hog1,
	ghog::lib::HogDescriptor* hog2,
	std::string hog_name1,
	std::string hog_name2,
	std::vector< std::string > image_list,
	cv::Size img_size,
	cv::Size window_size,
	int num_experiments,
	boost::random::mt19937 random_gen);

#endif /* DIFFERENCE_H_ */
