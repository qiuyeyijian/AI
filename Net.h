#pragma once

#include<vector>
#include "Config.h"

using std::vector;

// 样本类
class Sample {
public:
	vector<double> feature, label;	// 样本的特征数据和标签
public:
	Sample();
	Sample(const vector<double>& feature, const vector<double>& label);
	void display();

};

class Node {
public:
	double value, bias, bias_delta;
	vector<double> weight, weight_delta;

public:
	Node();
	// 加上explicit表示不能进行隐式类型转换
	explicit Node(int nextLayerSize);
};

class Net {
private:
	Node* inputLayer[Config::INNODE];
	Node* hiddenLayer[Config::HIDENODE];
	Node* outputLayer[Config::OUTNODE];

private:
	/**
	* Clear all gradient accumulation
	* set 'weight_delta'(the weight correction value)
	* and 'bias_delta'(the bias correction value) to 0 of nodes
	*/
	void grad_zero();

	// Forward propagation
	void forward();

	// Calculate the value of loss
	double lossFunc(const vector<double>& label);

	// Back propagation
	void backward(const vector<double>& label);

	// Revise 'weight' and 'bias' according to 'weight_delta' and 'bias_delta'
	void revise(size_t batch_size);

public:
	Net();

	// Training network
	bool train(const vector<Sample>& trainDataSet);

	// Using network to predict sample
	Sample predict(const vector<double>& feature);

	// Using network to predict the sample set
	vector<Sample> predict(const vector<Sample>& predictDataSet);

};
