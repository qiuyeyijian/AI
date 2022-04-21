#include"Net.h"
#include "Utils.h"
#include<random>



Net::Net() {
	std::mt19937 rd(std::random_device{}());
	std::uniform_real_distribution<double> distri(-1, 1);

	// Initialize input layer
	for (int i = 0; i < Config::INNODE; ++i) {
		// 创建一个输入层节点，同时指明隐藏层有多少个节点
		inputLayer[i] = new Node(Config::HIDENODE);

		// 初始化该节点所发出的所有权值
		for (int j = 0; j < Config::HIDENODE; ++j) {
			inputLayer[i]->weight[j] = distri(rd);

			inputLayer[i]->weight_delta[j] = 0.f;
		}
	}

	// Initialize hidden layer
	for (int i = 0; i < Config::HIDENODE; ++i) {
		// 创建一个隐藏层节点，同时指明输出层有多少个节点
		hiddenLayer[i] = new Node(Config::OUTNODE);

		// 一个节点对应一个偏置值，所以直接进行初始化
		hiddenLayer[i]->bias = distri(rd);
		hiddenLayer[i]->bias_delta = 0.f;

		for (int j = 0; j < Config::OUTNODE; ++j) {
			hiddenLayer[i]->weight[j] = distri(rd);
			hiddenLayer[i]->weight_delta[j] = 0.f;
		}
	}

	// Initialize output layer
	for (int i = 0; i < Config::OUTNODE; ++i) {
		// 创建一个输出层节点
		outputLayer[i] = new Node(0);

		outputLayer[i]->bias = distri(rd);
		outputLayer[i]->bias_delta = 0.f;
	}

}

void Net::grad_zero() {
	// 输入层的所有weight_delta清零
	for (auto& x : inputLayer) {
		x->weight_delta.assign(x->weight_delta.size(), 0.f);
	}

	// 隐藏层的所有weight_delta，bias_delta清零
	for (auto& x : hiddenLayer) {
		x->bias_delta = 0.f;
		x->weight_delta.assign(x->weight_delta.size(), 0.f);
	}

	// 输出层的所有bias_delta清零
	for (auto& x : outputLayer) {
		x->bias_delta = 0.f;
	}
}

void Net::forward() {
	// 从输入层传播到隐藏层
	// 对于每一个隐藏层节点，遍历所有输入层节点
	for (int i = 0; i < Config::HIDENODE; ++i) {
		double sum = 0;

		// 所有输入层节点的值乘以对应权值，然后求和
		for (int j = 0; j < Config::INNODE; ++j) {
			sum += inputLayer[j]->value * inputLayer[j]->weight[i];
		}
		// 加上或者减去偏置值都行，这里选择减去
		sum -= hiddenLayer[i]->bias;

		hiddenLayer[i]->value = Utils::sigmoid(sum);
	}

	// 从隐藏层传播到输出层
	// 对于每一个输出层节点，遍历所有隐藏层节点
	for (int i = 0; i < Config::OUTNODE; ++i) {
		double sum = 0;

		for (int j = 0; j < Config::HIDENODE; ++j) {
			sum += hiddenLayer[j]->value * hiddenLayer[j]->weight[i];
		}
		sum -= outputLayer[i]->bias;

		outputLayer[i]->value = Utils::sigmoid(sum);
	}
}

double Net::lossFunc(const vector<double>& label) {
	double loss = 0.f;

	for (int i = 0; i < Config::OUTNODE; ++i) {
		loss += std::pow(outputLayer[i]->value - label[i], 2) / 2;
	}

	return loss;
}

void Net::backward(const vector<double>& label) {
	// 计算输出层节点的bias_delta
	for (int i = 0; i < Config::OUTNODE; ++i) {
		double bias_delta = -(label[i] - outputLayer[i]->value) *
			outputLayer[i]->value * (1.0 - outputLayer[i]->value);

		outputLayer[i]->bias_delta += bias_delta;
	}

	// 计算隐藏层节点的weight_delta
	// 对于每一个隐藏层节点，遍历所有输出层节点
	for (int i = 0; i < Config::HIDENODE; ++i) {
		for (int j = 0; j < Config::OUTNODE; ++j) {
			double weight_delta = (label[j] - outputLayer[j]->value) *
				outputLayer[j]->value * (1.0 - outputLayer[j]->value) *
				hiddenLayer[i]->value;

			hiddenLayer[i]->weight_delta[j] += weight_delta;
		}
	}

	// 计算隐藏层的bias_delta
	for (int i = 0; i < Config::HIDENODE; ++i) {
		double bias_delta = 0.f;
		for (int j = 0; j < Config::OUTNODE; ++j) {
			bias_delta += -(label[j] - outputLayer[j]->value) *
				outputLayer[j]->value * (1.0 - outputLayer[j]->value) *
				hiddenLayer[i]->weight[j];
		}

		bias_delta *= hiddenLayer[i]->value * (1.0 - hiddenLayer[i]->value);

		hiddenLayer[i]->bias_delta += bias_delta;
	}

	// 计算输入层的weight_delta
	for (int i = 0; i < Config::INNODE; ++i) {
		for (int j = 0; j < Config::HIDENODE; ++j) {
			double weight_delta = 0.f;

			for (int k = 0; k < Config::OUTNODE; ++k) {
				weight_delta += (label[k] - outputLayer[k]->value) *
					outputLayer[k]->value * (1.0 - outputLayer[k]->value) *
					hiddenLayer[j]->weight[k];
			}

			weight_delta *= hiddenLayer[j]->value * (1.0 - hiddenLayer[j]->value) *
				inputLayer[i]->value;

			inputLayer[i]->weight_delta[j] += weight_delta;
		}
	}
}


bool Net::train(const vector<Sample>& trainDataSet) {
	for (int epoch = 0; epoch <= Config::max_epoch; ++epoch) {
		grad_zero();
		double max_loss = 0.f;

		for (const Sample& sample : trainDataSet) {

			for (int i = 0; i < Config::INNODE; ++i) {
				inputLayer[i]->value = sample.feature[i];
			}

			forward();

			double loss = lossFunc(sample.label);
			max_loss = std::max(max_loss, loss);

			backward(sample.label);
		}

		if (max_loss < Config::threshold) {
			printf("Training success in %lu epochs. \n", epoch);
			printf("Final maximum error(loss): %lf\n", max_loss);
			return true;
		}
		else if (epoch % 5000 == 0) {
			printf("#epoch %-7lu - max_loss: %lf\n", epoch, max_loss);
		}

		revise(trainDataSet.size());

	}

	printf("Failed within %lu epoch.", Config::max_epoch);

	return false;
}

void Net::revise(size_t batch_size) {
	for (int i = 0; i < Config::INNODE; ++i) {
		for (int j = 0; j < Config::HIDENODE; ++j) {
			inputLayer[i]->weight[j] += Config::lr * inputLayer[i]->weight_delta[j] / (double)batch_size;
		}
	}

	for (int i = 0; i < Config::HIDENODE; ++i) {
		hiddenLayer[i]->bias += Config::lr * hiddenLayer[i]->bias_delta / (double)batch_size;

		for (int j = 0; j < Config::OUTNODE; ++j) {
			hiddenLayer[i]->weight[j] += Config::lr * hiddenLayer[i]->weight_delta[j] / (double)batch_size;
		}
	}

	for (int i = 0; i < Config::OUTNODE; ++i) {
		outputLayer[i]->bias += Config::lr * outputLayer[i]->bias_delta / (double)batch_size;
	}

}


Sample Net::predict(const vector<double>& feature) {
	for (int i = 0; i < Config::INNODE; ++i) {
		inputLayer[i]->value = feature[i];
	}

	forward();

	vector<double> label(Config::OUTNODE);
	for (int i = 0; i < Config::OUTNODE; ++i) {
		label[i] = outputLayer[i]->value;
	}

	return Sample(feature, label);
}


vector<Sample> Net::predict(const vector<Sample>& predictDataSet) {
	vector<Sample> predSet;

	for (auto& sample : predictDataSet) {
		predSet.push_back(predict(sample.feature));
	}

	return predSet;
}


Node::Node() : value(0.f), bias(0.f), bias_delta(0.f) {}
Node::Node(int nextLayerSize) : value(0.f), bias(0.f), bias_delta(0.f) {
	weight.resize(nextLayerSize);
	weight_delta.resize(nextLayerSize);
}

Sample::Sample() = default;
Sample::Sample(const vector<double>&feature, const vector<double>&label) : feature(feature), label(label) {}

void Sample::display() {
	printf("\ninput: ");
	for (const auto& x : feature) printf("%lf ", x);

	printf("\noutput: ");
	for (const auto& x : label) printf("%lf ", x);
	printf("\n");
}