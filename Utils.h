#pragma once

#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>
#include<string>

#include "Config.h"

using std::vector;
using std::string;
// ���ǰ�����������ͷ�ļ��ظ��������⣬�����������ͷ�ļ�����.cpp�ļ��а���
class Sample;

namespace Utils {
	static double sigmoid(double x) {
		return 1.0 / (1.0 + std::exp(-x));
	}

	vector<double> getFileData(const string& filename);

	vector<Sample> getTrainData(const string& filename);

	vector<Sample> getTestData(const string& filename);
}

