#include "stats.h"

#include <cmath>
#include <iostream>
#include <cfloat>

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

double report_statistics(std::vector< double > values,
	int ratio,
	std::string unit)
{
	std::vector< double > results = calculate_statistics(values);

	std::cout << "Average: " << results[0] * ratio << " " << unit << std::endl;
	std::cout << "Standard deviation: " << results[1] * ratio << std::endl;
	std::cout << "Max: " << results[2] * ratio << " " << unit << std::endl;
	std::cout << "Min: " << results[3] * ratio << " " << unit << std::endl;
	std::cout << std::endl;
	return results[0];
}
