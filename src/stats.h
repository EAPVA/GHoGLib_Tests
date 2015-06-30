/*
 * stats.h
 *
 *  Created on: Jun 29, 2015
 *      Author: teider
 */

#ifndef STATS_H_
#define STATS_H_

#include <vector>
#include <string>

std::vector< double > calculate_statistics(std::vector< double > values);

double report_statistics(std::vector< double > values,
	int ratio,
	std::string unit);

#endif /* STATS_H_ */
