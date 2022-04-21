#pragma once

#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>
#include<string>

#include "Config.h"

using std::vector;
using std::string;
// 类的前置声明，解决头文件重复包含问题，将引起问题的头文件放在.cpp文件中包含
class Sample;

namespace Utils {
	static double sigmoid(double x) {
		return 1.0 / (1.0 + std::exp(-x));
	}

	vector<double> getFileData(const string& filename);

	vector<Sample> getTrainData(const string& filename);

	vector<Sample> getTestData(const string& filename);
}

