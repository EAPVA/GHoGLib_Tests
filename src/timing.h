/*
 * timing.h
 *
 *  Created on: Jun 29, 2015
 *      Author: teider
 */

#ifndef TIMING_H_
#define TIMING_H_

#include <string>

#include <boost/random/mersenne_twister.hpp>

#include <opencv2/core/core.hpp>

#include <include/HogDescriptor.h>

double measure_time(ghog::lib::HogDescriptor* hog,
	std::string hog_name,
	std::vector< std::string > image_list,
	cv::Size img_size,
	int num_experiments,
	boost::random::mt19937 random_gen);

double measure_time_opencv(std::vector< std::string > image_list,
	cv::Size img_size,
	int num_experiments,
	boost::random::mt19937 random_gen);

#endif /* TIMING_H_ */
