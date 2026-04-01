#include <iostream>
#include <print>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <cmath>


template<typename T>
using matrix = std::vector<std::vector<T>>;


// 矩阵乘法
matrix<double> matrixMultiply(const matrix<double>& a, const matrix<double>& b)
{
	int widthA = a[0].size(), heightA = a.size(), widthB = b[0].size(), heightB = b.size();
	matrix<double> res(heightA, std::vector<double>(widthB, 0.0));
	if (widthA == heightB)
	{
		for (int i = 0; i < heightA; i++)
		{
			for (int k = 0; k < widthA; k++)
			{
				for (int j = 0; j < widthB; j++)
				{
					res[i][j] += (a[i][k] * b[k][j]);
				}
			}
		}
		return res;
	}
	else
	{
		throw std::runtime_error("矩阵维度不匹配：A 的列数必须等于 B 的行数");
	}
}

std::vector<double> matrixMultiply(const matrix<double>& weight, const std::vector<double>& input)
{
	int  heightInput = input.size(), widthWeight = weight[0].size(), heightWeight = weight.size();
	std::vector<double> res(heightWeight, 0.0);
	if (heightInput == widthWeight)
	{
		for (int i = 0; i < heightWeight; i++)
		{
			for (int k = 0; k < widthWeight; k++)
			{
				res[i] += (input[k] * weight[i][k]);
			}
		}
		return res;
	}
	else
	{
		throw std::runtime_error("矩阵维度不匹配：A 的列数必须等于 B 的行数");
	}
}

std::vector<double> matrixAdd(const std::vector<double>& a, const std::vector<double>& b)
{
	int height = a.size();
	std::vector<double> res(height, 0);
	for (int i = 0; i < height; i++)
	{
		res[i] = a[i] + b[i];
	}
	return res;
}

double ReLU(double x)
{
	return std::max(0.0, x);
}

double dReLU(double x)
{
	return (x > 0 ? 1 : 0);
}

std::vector<double> ReLU(const std::vector<double>& x)
{
	std::vector<double>res = x;
	for (double& i : res)
	{
		i = ReLU(i);
	}
	return res;
}

std::vector<double> dReLU(const std::vector<double>& x)
{
	std::vector<double>res = x;
	for (double& i : res)
	{
		i = dReLU(i);
	}
	return res;
}

void randomizeMatrix(matrix<double>& a)
{
	int height = a.size();
	int width = a[0].size();

	std::random_device rd;
	std::mt19937 gen(rd());

	double limit = sqrt(6.0 / width);

	std::uniform_real_distribution<double> dist(-limit, limit);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			a[i][j] = dist(gen);
		}
	}
}

std::vector<double> NeuroCalc(const std::vector<double>& input, const std::vector<matrix<double>>& weightMatrixs, const std::vector<std::vector<double>>& biasMatrixs)
{
	int depth = biasMatrixs.size();

	std::vector<double> res = input;
	for (int i = 0; i < depth - 1; i++)
	{
		res = ReLU(matrixAdd(matrixMultiply(weightMatrixs[i], res), biasMatrixs[i]));
	}

	res = matrixAdd(matrixMultiply(weightMatrixs[depth - 1], res), biasMatrixs[depth - 1]);

	return res;
}


std::vector<double> NeuroCalc(
	const std::vector<double>& input,
	const std::vector<matrix<double>>& weightMatrixs,
	const std::vector<std::vector<double>>& biasMatrixs)
{
	int depth = biasMatrixs.size();

	std::vector<double> res = input;
	for (int i = 0; i < depth - 1; i++)
	{
		res = ReLU(matrixAdd(matrixMultiply(weightMatrixs[i], res), biasMatrixs[i]));
	}

	res = matrixAdd(matrixMultiply(weightMatrixs[depth - 1], res), biasMatrixs[depth - 1]);

	return res;
}

std::vector<double> NeuroCalc(
	const std::vector<double>& input,
	const std::vector<matrix<double>>& weightMatrixs,
	const std::vector<std::vector<double>>& biasMatrixs,
	std::vector<std::vector<double>>& rawInput,
	std::vector<std::vector<double>>& Input)
{
	int depth = biasMatrixs.size();

	std::vector<double> res = input;
	for (int i = 0; i < depth - 1; i++)
	{
		rawInput[i] = matrixAdd(matrixMultiply(weightMatrixs[i], res), biasMatrixs[i]);
		Input[i] = ReLU(rawInput[i]);
		res = Input[i];
	}

	rawInput[depth - 1] = matrixAdd(matrixMultiply(weightMatrixs[depth - 1], res), biasMatrixs[depth - 1]);
	Input[depth - 1] = ReLU(rawInput[depth - 1]);
	res = Input[depth - 1];

	return res;
}


double loss(std::vector<double> NeuroNet_output, std::vector<double> train_output)
{
	double loss = 0;
	int length = NeuroNet_output.size();
	for (int i = 0; i < length; i++)
	{
		loss += pow(NeuroNet_output[i] - train_output[i], 2);
	}
	return loss;
}


void backPropagate(std::vector<double> trainInput, std::vector<double> neuroOutput, std::vector<double> rightOutput)
{
	NeuroCalc(trainInput, weightMatrixs, biasMatrixs, rawInput, Input);

	int l = notOutputLayers - 1;

	int height = weightDiffMatrixs[l].size();
	int width = weightDiffMatrixs[l][0].size();

	double errReuse;

	for (int i = 0; i < height; i++)
	{

		double err = (2.0 / outputHeight) * (neuroOutput[i] - rightOutput[i]);
		biasMatrixs[l][i] = err;

		for (int j = 0; j < width; j++)
		{
			weightDiffMatrixs[l][i][j] = err * Input[l][j];
		}
	}




	for (int l = notOutputLayers - 2; l >= 0; l--)
	{

	}

}















std::vector<int> layerWidths = { 6,64,32,1 };
int notOutputLayers = layerWidths.size() - 1;
int outputHeight = layerWidths[notOutputLayers]; // 这里取巧了, 数值恰好能复用而已

// weightMatrix[层][行][列]
// weightMatrix[0], 是一个二维


std::vector<matrix<double>> weightMatrixs(notOutputLayers);
std::vector<std::vector<double>> biasMatrixs(notOutputLayers);

std::vector<matrix<double>> weightDiffMatrixs(notOutputLayers);
std::vector<std::vector<double>> biasDiffMatrixs(notOutputLayers);


std::vector<std::vector<double>> rawInput(notOutputLayers);
std::vector<std::vector<double>> Input(notOutputLayers); // 这里貌似需要 -1 ?

int main()
{

	for (int layer = 0; layer < notOutputLayers; layer++)
	{
		biasMatrixs[layer].resize(layerWidths[layer + 1]);
		biasDiffMatrixs[layer].resize(layerWidths[layer + 1]);

		weightMatrixs[layer].resize(layerWidths[layer + 1]);
		weightDiffMatrixs[layer].resize(layerWidths[layer + 1]);

		rawInput[layer].resize(layerWidths[layer]);
		Input[layer].resize(layerWidths[layer]);
		for (int i = 0; i < layerWidths[layer + 1]; i++)
		{
			weightMatrixs[layer][i].resize(layerWidths[layer]);
			weightDiffMatrixs[layer][i].resize(layerWidths[layer]);
		}
		randomizeMatrix(weightMatrixs[layer]);
	}

	std::vector<double> ans(1, 0);
	std::vector<double> input = { 2,3,1,0,0,0 };
	ans = NeuroCalc(input, weightMatrixs, biasMatrixs);
	std::print("{}", ans);








	// 此处是训练集
	matrix<double> trainMatrixs_input(100, std::vector<double>(6, 0));
	matrix<double> trainMatrixs_output(100, std::vector<double>(1, 0));
	// [ [num1], [num2], [+], [-], [*], [/] ]
	// [ [ans] ]

	std::random_device rd;
	std::mt19937 gen(rd());

	std::uniform_real_distribution<double> dist(-10, 10);

	for (int i = 0; i < 100; i++)
	{
		double num1 = dist(gen);
		double num2 = dist(gen);
		double op = dist(gen);
		double ans;

		trainMatrixs_input[i][0] = num1;
		trainMatrixs_input[i][1] = num2;

		if (op < -5)
		{
			trainMatrixs_input[i][2] = 1;
			ans = num1 + num2;
		}
		else if (op < 0)
		{
			trainMatrixs_input[i][3] = 1;
			ans = num1 - num2;
		}
		else if (op < 5)
		{
			trainMatrixs_input[i][4] = 1;
			ans = num1 * num2;
		}
		else
		{
			trainMatrixs_input[i][5] = 1;
			ans = num1 / num2;
		}
		trainMatrixs_output[i][0] = ans;

	}






}


