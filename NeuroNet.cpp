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

std::vector<double> ReLU(const std::vector<double>& x)
{
	std::vector<double>res = x;
	for (double& i : res)
	{
		i = ReLU(i);
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





















std::vector<int> layerWidths = { 6,64,32,1 };
int notOutputLayers = layerWidths.size() - 1;


// weightMatrix[层][行][列]
// weightMatrix[0], 是一个二维


std::vector<matrix<double>> weightMatrixs(notOutputLayers);
std::vector<std::vector<double>> biasMatrixs(notOutputLayers);




int main()
{

	for (int layer = 0; layer < notOutputLayers; layer++)
	{
		biasMatrixs[layer].resize(layerWidths[layer + 1]);

		weightMatrixs[layer].resize(layerWidths[layer + 1]);
		for (int i = 0; i < layerWidths[layer + 1]; i++)
		{
			weightMatrixs[layer][i].resize(layerWidths[layer]);
		}
		randomizeMatrix(weightMatrixs[layer]);
	}

	std::vector<double> ans(1, 0);
	std::vector<double> input = { 2,3,1,0,0,0 };
	ans = NeuroCalc(input, weightMatrixs, biasMatrixs);
	std::print("{}", ans);


}


