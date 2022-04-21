#include<iostream>
#include "Utils.h"
#include "Net.h"

using std::cout;

int main() {
	// Create neural network
	Net net;

	// Read in training data
	const vector<Sample> trainDataSet = Utils::getTrainData("train.txt");

	// Training
	net.train(trainDataSet);

	// Predict
	const vector<Sample> testDataSet = Utils::getTestData("test.txt");
	vector<Sample> predSet = net.predict(testDataSet);

	for (auto& pred : predSet) {
		pred.display();
	}

	return 0;
}