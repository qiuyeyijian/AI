#include"Net.h"
#include "Utils.h"
#include<random>



Net::Net() {
	std::mt19937 rd(std::random_device{}());
	std::uniform_real_distribution<double> distri(-1, 1);

	// Initialize input layer
	for (int i = 0; i < Config::INNODE; ++i) {
		// ����һ�������ڵ㣬ͬʱָ�����ز��ж��ٸ��ڵ�
		inputLayer[i] = new Node(Config::HIDENODE);

		// ��ʼ���ýڵ�������������Ȩֵ
		for (int j = 0; j < Config::HIDENODE; ++j) {
			inputLayer[i]->weight[j] = distri(rd);

			inputLayer[i]->weight_delta[j] = 0.f;
		}
	}

	// Initialize hidden layer
	for (int i = 0; i < Config::HIDENODE; ++i) {
		// ����һ�����ز�ڵ㣬ͬʱָ��������ж��ٸ��ڵ�
		hiddenLayer[i] = new Node(Config::OUTNODE);

		// һ���ڵ��Ӧһ��ƫ��ֵ������ֱ�ӽ��г�ʼ��
		hiddenLayer[i]->bias = distri(rd);
		hiddenLayer[i]->bias_delta = 0.f;

		for (int j = 0; j < Config::OUTNODE; ++j) {
			hiddenLayer[i]->weight[j] = distri(rd);
			hiddenLayer[i]->weight_delta[j] = 0.f;
		}
	}

	// Initialize output layer
	for (int i = 0; i < Config::OUTNODE; ++i) {
		// ����һ�������ڵ�
		outputLayer[i] = new Node(0);

		outputLayer[i]->bias = distri(rd);
		outputLayer[i]->bias_delta = 0.f;
	}

}

void Net::grad_zero() {
	// ����������weight_delta����
	for (auto& x : inputLayer) {
		x->weight_delta.assign(x->weight_delta.size(), 0.f);
	}

	// ���ز������weight_delta��bias_delta����
	for (auto& x : hiddenLayer) {
		x->bias_delta = 0.f;
		x->weight_delta.assign(x->weight_delta.size(), 0.f);
	}

	// ����������bias_delta����
	for (auto& x : outputLayer) {
		x->bias_delta = 0.f;
	}
}

void Net::forward() {
	// ������㴫�������ز�
	// ����ÿһ�����ز�ڵ㣬�������������ڵ�
	for (int i = 0; i < Config::HIDENODE; ++i) {
		double sum = 0;

		// ���������ڵ��ֵ���Զ�ӦȨֵ��Ȼ�����
		for (int j = 0; j < Config::INNODE; ++j) {
			sum += inputLayer[j]->value * inputLayer[j]->weight[i];
		}
		// ���ϻ��߼�ȥƫ��ֵ���У�����ѡ���ȥ
		sum -= hiddenLayer[i]->bias;

		hiddenLayer[i]->value = Utils::sigmoid(sum);
	}

	// �����ز㴫���������
	// ����ÿһ�������ڵ㣬�����������ز�ڵ�
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
	// ���������ڵ��bias_delta
	for (int i = 0; i < Config::OUTNODE; ++i) {
		double bias_delta = -(label[i] - outputLayer[i]->value) *
			outputLayer[i]->value * (1.0 - outputLayer[i]->value);

		outputLayer[i]->bias_delta += bias_delta;
	}

	// �������ز�ڵ��weight_delta
	// ����ÿһ�����ز�ڵ㣬�������������ڵ�
	for (int i = 0; i < Config::HIDENODE; ++i) {
		for (int j = 0; j < Config::OUTNODE; ++j) {
			double weight_delta = (label[j] - outputLayer[j]->value) *
				outputLayer[j]->value * (1.0 - outputLayer[j]->value) *
				hiddenLayer[i]->value;

			hiddenLayer[i]->weight_delta[j] += weight_delta;
		}
	}

	// �������ز��bias_delta
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

	// ����������weight_delta
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