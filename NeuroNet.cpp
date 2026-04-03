#include <iostream>
#include <print>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <cmath>
#include <string>
#include <sstream>


template<typename T>
using matrix = std::vector<std::vector<T>>;



double learnRate = 0.01;

std::vector<int> layerWidths = { 6,64,64,32,1 }; // 输入, ..., 输出
int notOutputLayers = layerWidths.size() - 1;
int outputHeight = layerWidths[layerWidths.size() - 1];

// weightMatrix[层][行][列]
// weightMatrix[0], 是一个二维


std::vector<matrix<double>> weightMatrixs(notOutputLayers);
std::vector<std::vector<double>> biasMatrixs(notOutputLayers);

std::vector<matrix<double>> weightDiffMatrixs(notOutputLayers);
std::vector<std::vector<double>> biasDiffMatrixs(notOutputLayers);


std::vector<std::vector<double>> rawInput(notOutputLayers);
std::vector<std::vector<double>> Input(notOutputLayers);


// 假设是第 l 层
//
//
// ... ----------->				  Input[l]									       ------------------------->   Input[l+1]		
//              weightMatrixs[l]           +   biasMatrixs[l]   ->  rawInput[l] ---		     weightMatrixs[l]                 +   ....
//
//




// 矩阵乘法

// AB
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


// Ax
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
std::vector<double> matrixMinus(const std::vector<double>& a, const std::vector<double>& b)
{
	int height = a.size();
	std::vector<double> res(height, 0);
	for (int i = 0; i < height; i++)
	{
		res[i] = a[i] - b[i];
	}
	return res;


}
matrix<double> matrixMinus(const matrix<double>& a, const matrix<double>& b)
{
	int height = a.size();
	int width = a[0].size();
	matrix<double> res(height, std::vector<double>(width, 0));
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			res[i][j] = a[i][j] - b[i][j]
				;
		}
	}
	return res;
}
std::vector<double> matrixMultiply_Num(double a, const std::vector<double>& b)
{
	int height = b.size();
	std::vector<double> res(height, 0);
	for (int i = 0; i < height; i++)
	{
		res[i] = b[i] * a;
	}
	return res;
}


matrix<double> matrixMultiply_Num(double a, const matrix<double>& b)
{
	int height = b.size();
	int width = b[0].size();

	matrix<double> res(height, std::vector<double>(width, 0));
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			res[i][j] = a * b[i][j];
		}
	}
	return res;
}

matrix<double> outerProduct(const std::vector<double>& a, const std::vector<double>& b)
{
	matrix<double> res(a.size(), std::vector<double>(b.size(), 0));
	for (int i = 0; i < a.size(); i++)
	{
		for (int j = 0; j < b.size(); j++)
		{
			res[i][j] = a[i] * b[j];
		}
	}
	return res;
}

//matrix<double> outerProduct(const std::vector<double>& a, const std::vector<double>& b)
//{
//	matrix<double> res(a.size(), std::vector<double>(b.size() - 1, 0));
//	for (int i = 0; i < a.size(); i++)
//	{
//		for (int j = 0; j < b.size(); j++)
//		{
//			res[i][j] = a[i] * b[j];
//		}
//	}
//	return res;
//}


std::vector<double> hadamardProduct(const std::vector<double>& a, const std::vector<double>& b)
{
	std::vector<double> res(a.size(), 0);
	for (int i = 0; i < a.size(); i++)
	{
		res[i] = a[i] * b[i];
	}
	return res;
}




matrix<double> transpose(const matrix<double>& x)
{

	int height = x[0].size();
	int width = x.size();
	matrix<double> res(height, std::vector<double>(width, 0));
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			res[j][i] = x[i][j];
		}
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
	Input[depth - 1] = rawInput[depth - 1];
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


void backPropagate(std::vector<double> trainInput, std::vector<double> rightOutput)
{

	NeuroCalc(trainInput, weightMatrixs, biasMatrixs, rawInput, Input);


	std::vector<double> delta = matrixMultiply_Num((2.0 / layerWidths.back()), matrixMinus(rawInput.back(), rightOutput));

	int last = layerWidths.size() - 2;
	const std::vector<double>& prevActLast = (last == 0) ? trainInput : Input[last - 1];


	weightDiffMatrixs[last] = outerProduct(delta, prevActLast);


	biasDiffMatrixs[last] = delta;



	for (int i = last - 1; i >= 0; i--)
	{

		// 求 delta_i
		delta = hadamardProduct(matrixMultiply(transpose(weightMatrixs[i + 1]), delta), dReLU(rawInput[i]));

		const std::vector<double>& prevAct = (i == 0) ? trainInput : Input[i - 1];


		weightDiffMatrixs[i] = outerProduct(delta, prevAct);


		biasDiffMatrixs[i] = delta;





	}

	for (int i = last; i >= 0; i--)
	{

		weightMatrixs[i] = matrixMinus(weightMatrixs[i], matrixMultiply_Num(learnRate, weightDiffMatrixs[i]));

		biasMatrixs[i] = matrixMinus(biasMatrixs[i], matrixMultiply_Num(learnRate, biasDiffMatrixs[i]));
	}





	/*int l = notOutputLayers - 1;

	int height = weightDiffMatrixs[l].size();
	int width = weightDiffMatrixs[l][0].size();



	for (int i = 0; i < height; i++)
	{

		double err = (2.0 / outputHeight) * (neuroOutput[i] - rightOutput[i]);
		biasMatrixs[l][i] = err;

		for (int j = 0; j < width; j++)
		{
			weightDiffMatrixs[l][i][j] = err * Input[l][j];
		}
	}*/
	/*for (int l = notOutputLayers - 2; l >= 0; l--)
	{

	}*/

}


int countDigits(int n) {
	if (n == 0) return 1;   // 0 有一位
	int count = 0;
	n = std::abs(n);        // 处理负数
	while (n > 0) {
		n /= 10;
		++count;
	}
	return count;
}




// 自定义函数 自定义函数 自定义函数 自定义函数 自定义函数 自定义函数 自定义函数 自定义函数





// 这是一个把输入形如 "2*3" 的字符串转化为 [2, 3, 0, 0, 1, 0] 的函数
std::vector<double> stringToVector(std::string input)
{
	std::vector<double> res(6, 0);
	std::stringstream ss(input);

	double num1, num2;
	char op;
	if (ss >> num1 >> op >> num2)
	{

		/*if (num1 > 10 || num1 < -10 || num2 > 10 || num2 < -10)
		{
			std::print("输入范围: [-10, 10]");
		}*/
		res[0] = num1;
		res[1] = num2;
		if (op == '+')
		{
			res[2] = 1; // 加法
		}
		else if (op == '-')
		{
			res[3] = 1; // 减法
		}
		else if (op == '*')
		{
			res[4] = 1; // 乘法
		}
		else if (op == '/')
		{
			res[5] = 1; // 除法
		}
		else
		{
			std::cout << "不支持的运算符" << std::endl;
		}
	}
	else
	{
		std::cout << "输入格式不正确" << std::endl;
	}

	return res;
}

double logNormalize(double x)
{
	double sign = (x >= 0.0 ? 1.0 : -1.0);

	return sign * (log(abs(x) + 1.0)) / log(100.0);

}


std::vector<double> logNormalize(const std::vector<double>& x)
{
	int height = x.size();
	std::vector<double> res(height);
	for (int i = 0; i < x.size(); i++)
	{
		double sign = (x[i] >= 0.0 ? 1.0 : -1.0);

		res[i] = sign * (log(abs(x[i]) + 1.0)) / log(100.0);
	}
	return res;

}

double inverseLogNormalize(double x)
{
	double sign = (x >= 0.0 ? 1.0 : -1.0);

	return  sign * (pow(100.0, abs(x)) - 1.0);
}


std::vector<double> inverseLogNormalize(const std::vector<double>& x)
{

	int height = x.size();
	std::vector<double> res(height);
	for (int i = 0; i < height; i++)
	{
		double sign = (x[i] >= 0.0 ? 1.0 : -1.0);

		res[i] = sign * (pow(100.0, abs(x[i])) - 1.0);
	}
	return res;
}

// 这是一个把 [2, 3, 0, 0, 1, 0] 变成 6 的函数
double clacVector(std::vector<double> input)
{
	if (input[2] == 1)
	{
		return input[0] + input[1];
	}
	else if (input[3] == 1)
	{
		return input[0] - input[1];
	}
	else if (input[4] == 1)
	{
		return input[0] * input[1];
	}
	if (input[5] == 1)
	{
		return input[0] / input[1];
	}
	return 0.0;
}


// 这是一个把 [2, 3, 0, 0, 1, 0] 的矩阵变成 [0.2, 0.3, 0, 0, 1, 0] 的函数
std::vector<double> vectorNorm(std::vector<double> input)
{
	input[0] /= 100;
	input[1] /= 100;
	return input;
}

// 这是一个提取输入矩阵的操作符的函数
char vectorToOp(std::vector<double> input)
{
	if (input[2] == 1)
	{
		return '+';
	}
	else if (input[3] == 1)
	{
		return '-';
	}
	else if (input[4] == 1)
	{
		return '*';
	}
	if (input[5] == 1)
	{
		return '/';
	}
	return '?';
}



int main()
{

	for (int layer = 0; layer < notOutputLayers; layer++)
	{
		biasMatrixs[layer].resize(layerWidths[layer + 1]);
		biasDiffMatrixs[layer].resize(layerWidths[layer + 1]);

		weightMatrixs[layer].resize(layerWidths[layer + 1]);
		weightDiffMatrixs[layer].resize(layerWidths[layer + 1]);

		rawInput[layer].resize(layerWidths[layer + 1]);
		Input[layer].resize(layerWidths[layer + 1]);
		for (int i = 0; i < layerWidths[layer + 1]; i++)
		{
			weightMatrixs[layer][i].resize(layerWidths[layer]);
			weightDiffMatrixs[layer][i].resize(layerWidths[layer]);
		}
		randomizeMatrix(weightMatrixs[layer]);
	}









	/*std::vector<double> ans(1, 0);
	std::vector<double> input = { 2,3,1,0,0,0 };
	ans = NeuroCalc(input, weightMatrixs, biasMatrixs);
	std::print("{}", ans);*/



	// 自由编辑

	int casePerEpoch = 1000;


	// 此处是训练集
	matrix<double> trainMatrixs_input(casePerEpoch, std::vector<double>(6, 0));
	matrix<double> trainMatrixs_output(casePerEpoch, std::vector<double>(1, 0));
	// [ [num1], [num2], [+], [-], [*], [/] ]
	// [ [ans] ]

	std::random_device rd;
	std::mt19937 gen(rd());

	std::uniform_real_distribution<double> dist(-100, 100);
	std::uniform_int_distribution<int> opDist(0, 3);
	std::print("开始训练\n");

	// 隐藏光标
	std::cout << "\033[?25l";

	int epoch = 100;


	int casePerEpochDigits = countDigits(casePerEpoch);
	int epochDigits = countDigits(epoch);



	for (int i = 0; i < epoch; i++)
	{

		// 每轮训练集 casePerEpoch 个
		for (int j = 0; j < casePerEpoch; j++)
		{
			double num1 = dist(gen);
			double num2 = dist(gen);
			int op = opDist(gen);
			double ans;

			trainMatrixs_input[j][0] = num1;
			trainMatrixs_input[j][1] = num2;

			// 在生成每个样本时，先清空操作符 one-hot
			std::fill(trainMatrixs_input[j].begin() + 2, trainMatrixs_input[j].end(), 0.0);

			if (op == 0)
			{
				trainMatrixs_input[j][2] = 1;
				ans = num1 + num2;
			}
			else if (op == 1)
			{
				trainMatrixs_input[j][3] = 1;
				ans = num1 - num2;
			}
			else if (op == 2)
			{
				trainMatrixs_input[j][4] = 1;
				ans = num1 * num2;
			}
			else
			{
				while (num2 >= -1 && num2 <= 1)
				{
					num2 = dist(gen);
				}
				trainMatrixs_input[j][1] = num2;
				trainMatrixs_input[j][5] = 1;
				ans = num1 / num2;
			}
			trainMatrixs_output[j][0] = ans;




		}




		for (int j = 0; j < casePerEpoch; j++)
		{
			// std::vector<double> neuroOutput = NeuroCalc(trainMatrixs_input[j], weightMatrixs, biasMatrixs);

			backPropagate(vectorNorm(trainMatrixs_input[j]), logNormalize(trainMatrixs_output[j]));
			std::print("\r进度: epoch: {: >{}} / {}, case: {: >{}} / {}\t\t", i + 1, epochDigits, epoch, j + 1, casePerEpochDigits, casePerEpoch);
		}


		// std::print("\r进度: epoch: {} / {}, case: {} / {}		", i + 1, epoch, 0, casePerEpoch);
	}



	std::print("\n完成!\n");

	// 显示光标
	std::cout << "\033[?25h";

	while (1)
	{
		std::print("input: ");
		std::string input;
		std::getline(std::cin, input);  // 读取整行
		std::vector<double> inputVec = stringToVector(input);


		std::vector<double> neuroAnsAfter = inverseLogNormalize(NeuroCalc(vectorNorm(inputVec), weightMatrixs, biasMatrixs));

		double neuroRes = neuroAnsAfter[0];
		double rightAns = clacVector(inputVec);


		double delta = abs((rightAns - neuroRes) / rightAns) * 100;

		std::print("\033[1A\r{} {} {} = {} | {} {:.2f}%\n", inputVec[0], vectorToOp(inputVec), inputVec[1], neuroRes, rightAns, delta);



	}




}






