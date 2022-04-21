#include "Utils.h"
#include "Net.h"

#if defined(WIN64) || defined(_WIN64) || defined(WIN32) || defined(_WIN32)
#include<direct.h>
#else
#include<unistd.h>
#endif

vector<double> Utils::getFileData(const string& filename) {
	vector<double> res;
	double val;

	std::ifstream in(filename);
	if (in.is_open()) {
		while (!in.eof()) {
			in >> val;
			res.push_back(val);
		}
		in.close();
	}
	else {
		char path[256];
		bool flag = _getcwd(path, sizeof(path));	// 获取当前工作目录绝对路径
		printf("%s\\%s not found \n", path, filename.c_str());
		exit(1);
	}
	return res;
}

vector<Sample> Utils::getTrainData(const string& filename) {
	vector<Sample> trainDataSet;
	vector<double> buf = getFileData(filename);

	for (int i = 0; i < buf.size(); i += Config::INNODE + Config::OUTNODE) {
		Sample trainSample;
		// Read in training sample 'feature'
		for (int j = 0; j < Config::INNODE; ++j) {
			trainSample.feature.push_back(buf[i + j]);
		}
		// Read in training sample 'label'
		for (int j = 0; j < Config::OUTNODE; ++j) {
			trainSample.label.push_back(buf[i + Config::INNODE + j]);
		}
		// Add sample to the 'trainDataSet'
		trainDataSet.push_back(trainSample);
	}

	return trainDataSet;
}


vector<Sample> Utils::getTestData(const string& filename) {
	vector<Sample> testDataSet;

	vector<double> buf = getFileData(filename);

	for (int i = 0; i < buf.size(); i += Config::INNODE) {
		Sample testSample;
		// Read in test sample 'feature'
		for (int j = 0; j < Config::INNODE; ++j) {
			testSample.feature.push_back(buf[i + j]);
		}
		// Add sample to the 'testDataSet'
		testDataSet.push_back(testSample);
	}

	return testDataSet;
}