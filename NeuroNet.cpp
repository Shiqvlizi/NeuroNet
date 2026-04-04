#include <iostream>
#include <print>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <cmath>
#include <string>
#include <sstream>

// 定义矩阵模板
template<typename T>
using matrix = std::vector<std::vector<T>>;


// 学习率, 决定着参数更新的每一步大小
// 如果过大, 可能导致模型不收敛, 难以达到使 loss 函数很低的那个位置
// 如果过小, 可能导致模型收敛速度缓慢
double learnRate = 0.01;

// 这里决定着模型结构: 输入层, 隐藏层, ..., 隐藏层, 输出层
// 一般地, 层与层之间会有一层激活函数, 除了最后的隐藏层到输出层
std::vector<int> layerWidths = { 6,64,64,32,1 }; // 输入, ..., 输出
int notOutputLayers = layerWidths.size() - 1;
int outputHeight = layerWidths[layerWidths.size() - 1];

// weightMatrix[层][行][列]
// weightMatrix[0], 是一个二维

// 这里是会用到的矩阵
std::vector<matrix<double>> weightMatrixs(notOutputLayers);
std::vector<std::vector<double>> biasMatrixs(notOutputLayers);

std::vector<matrix<double>> weightDiffMatrixs(notOutputLayers);
std::vector<std::vector<double>> biasDiffMatrixs(notOutputLayers);


std::vector<std::vector<double>> rawInput(notOutputLayers);
std::vector<std::vector<double>> Input(notOutputLayers);



// !!!![存疑]!!!!
// 
// 假设是第 l 层
//
//
// ... ----------->				  Input[l]									       ------------------------->   Input[l+1]		
//              weightMatrixs[l]           +   biasMatrixs[l]   ->  rawInput[l] ---		     weightMatrixs[l]                 +   ....
//
//
// !!!![存疑]!!!!



// 矩阵乘法

// AB
// 矩阵和矩阵的标准乘法
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
// 矩阵和向量的乘法
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


// 用于两个向量的相加
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

// 用于两个向量的相减
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

// 用于两个矩阵的相减
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

// 用于向量的数乘
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

// 用于矩阵的数乘
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

// 两个向量的 "外积 (outer)", 不是 "叉乘 (cross)", 那是两个概念
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

// 用于两个向量的逐元素乘积, 在神经网络中一般用于激活函数的求导矩阵参与的运算
std::vector<double> hadamardProduct(const std::vector<double>& a, const std::vector<double>& b)
{
	std::vector<double> res(a.size(), 0);
	for (int i = 0; i < a.size(); i++)
	{
		res[i] = a[i] * b[i];
	}
	return res;
}



// 转置一个矩阵
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

// 输入 W, x, 输出 W^T * x
std::vector<double> matrixMultiplyTransposed(const matrix<double>& weight, const std::vector<double>& input)
{
	int rows = weight.size();
	int cols = weight[0].size();
	if (input.size() != rows)
	{
		throw std::runtime_error("矩阵维度不匹配：W^T * x 中 x 的长度必须等于 W 的行数");
	}

	std::vector<double> res(cols, 0.0);
	for (int i = 0; i < rows; i++)
	{
		double x = input[i];
		for (int j = 0; j < cols; j++)
		{
			res[j] += weight[i][j] * x;
		}
	}
	return res;
}

// 激活函数以及它们的求导
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


// 随机化一个矩阵
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



// 当前模型的输出
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


// 当前模型的输出, 但是同时输出层与层之间的 z 和 a
// z -> rawInput
// a -> Input
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

// 损失函数, 这里是 loss 是 神经元输出和标准输出的差的平方的平均值
double loss(const std::vector<double>& NeuroNet_output, const std::vector<double>& train_output)
{
	double loss = 0;
	int length = NeuroNet_output.size();
	for (int i = 0; i < length; i++)
	{
		loss += pow(NeuroNet_output[i] - train_output[i], 2);
	}
	return loss;
}

// 反向传播函数
void backPropagate(const std::vector<double>& trainInput, const std::vector<double>& rightOutput)
{
	// 先获取现在这个模型的 rawInput 和 Input, 因为之后梯度的计算要用
	NeuroCalc(trainInput, weightMatrixs, biasMatrixs, rawInput, Input);

	// 最后一个权重/偏置矩阵的索引
	int last = layerWidths.size() - 2;

	// 初始化 deltas 的大小
	// 每一层的梯度计算, 都要依赖于 delta, 同时 delta 也是递归的核心
	// delta[i] = ( W[i+1]^T * delta[i+1] ) $ ReLU' ( z[i] )
	// delta 就是 dL/dz
	// $ 表示逐个元素相乘 (至于为什么公式是这个, 数学, 矩阵微分之后的结果就是它)
	// 此公式就建立了后一层和前一层的 delta 关系
	// 然后
	// biasDiff[i] = delta[i]
	// weightDiff[i] = delta[i] * a[i-1]^T 
	std::vector<std::vector<double>> deltas(notOutputLayers);


	// 初始化 delta[L-1]
	deltas[last].resize(layerWidths.back());
	for (int i = 0; i < layerWidths.back(); i++)
	{
		deltas[last][i] = (2.0 / layerWidths.back()) * (rawInput.back()[i] - rightOutput[i]);
	}


	// 计算每个 delta
	for (int i = last - 1; i >= 0; i--)
	{
		deltas[i] = matrixMultiplyTransposed(weightMatrixs[i + 1], deltas[i + 1]);
		for (int j = 0; j < deltas[i].size(); j++)
		{
			deltas[i][j] *= dReLU(rawInput[i][j]);
		}
	}


	// 根据 delta 更新参数
	for (int i = last; i >= 0; i--)
	{
		const std::vector<double>& prevAct = (i == 0) ? trainInput : Input[i - 1];
		const std::vector<double>& delta = deltas[i];

		for (int r = 0; r < weightMatrixs[i].size(); r++)
		{
			for (int c = 0; c < weightMatrixs[i][r].size(); c++)
			{
				weightMatrixs[i][r][c] -= learnRate * delta[r] * prevAct[c];
			}
			biasMatrixs[i][r] -= learnRate * delta[r];
		}
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

std::vector<double> vectorNorm(const std::vector<double>& input);


// 自定义函数 自定义函数 自定义函数 自定义函数 自定义函数 自定义函数 自定义函数 自定义函数


// 要训练的范围
// [-limit, limit]
double limit= 100;

// 这是一个把输入形如 "2*3" 的字符串转化为 [2, 3, 0, 0, 1, 0] 的函数
std::vector<double> stringToVector(const std::string& input)
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
			// std::cout << "不支持的运算符" << std::endl;
		}
	}
	else
	{
		// std::cout << "输入格式不正确" << std::endl;
	}

	return res;
}

double logNormalize(double x)
{
	double sign = (x >= 0.0 ? 1.0 : -1.0);

	return sign * (log(abs(x) + 1.0)) / log(limit);

}


std::vector<double> logNormalize(const std::vector<double>& x)
{
	int height = x.size();
	std::vector<double> res(height);
	for (int i = 0; i < x.size(); i++)
	{
		double sign = (x[i] >= 0.0 ? 1.0 : -1.0);

		res[i] = sign * (log(abs(x[i]) + 1.0)) / log(limit);
	}
	return res;

}

double inverseLogNormalize(double x)
{
	double sign = (x >= 0.0 ? 1.0 : -1.0);

	return  sign * (pow(limit, abs(x)) - 1.0);
}


std::vector<double> inverseLogNormalize(const std::vector<double>& x)
{

	int height = x.size();
	std::vector<double> res(height);
	for (int i = 0; i < height; i++)
	{
		double sign = (x[i] >= 0.0 ? 1.0 : -1.0);

		res[i] = sign * (pow(limit, abs(x[i])) - 1.0);
	}
	return res;
}

double evaluateLoss(int caseCount)
{
	if (caseCount <= 0)
	{
		return 0.0;
	}

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dist(-limit, limit);
	std::uniform_int_distribution<int> opDist(0, 3);

	double totalLoss = 0.0;
	for (int i = 0; i < caseCount; i++)
	{
		double num1 = dist(gen);
		double num2 = dist(gen);
		int op = opDist(gen);

		std::vector<double> inputVec(6, 0.0);
		inputVec[0] = num1;
		inputVec[1] = num2;

		double ans = 0.0;
		if (op == 0)
		{
			inputVec[2] = 1;
			ans = num1 + num2;
		}
		else if (op == 1)
		{
			inputVec[3] = 1;
			ans = num1 - num2;
		}
		else if (op == 2)
		{
			inputVec[4] = 1;
			ans = num1 * num2;
		}
		else
		{
			while (num2 >= -1 && num2 <= 1)
			{
				num2 = dist(gen);
			}
			inputVec[1] = num2;
			inputVec[5] = 1;
			ans = num1 / num2;
		}

		std::vector<double> neuroAns = inverseLogNormalize(NeuroCalc(vectorNorm(inputVec), weightMatrixs, biasMatrixs));
		std::vector<double> rightAns(1, ans);

		double sampleLoss = 0.0;
		const std::size_t outSize = std::min(neuroAns.size(), rightAns.size());
		for (std::size_t j = 0; j < outSize; j++)
		{
			sampleLoss += std::pow(neuroAns[j] - rightAns[j], 2);
		}
		totalLoss += sampleLoss;
	}

	return totalLoss / caseCount;
}

// 这是一个把 [2, 3, 0, 0, 1, 0] 变成 6 的函数
double clacVector(const std::vector<double>& input)
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
std::vector<double> vectorNorm(const std::vector<double>& input)
{
	std::vector<double> res = input;
	res[0] /= limit;
	res[1] /= limit;
	return res;
}

// 这是一个提取输入矩阵的操作符的函数
char vectorToOp(const std::vector<double>& input)
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

	int casePerEpoch = 20000;


	// 此处是训练集
	matrix<double> trainMatrixs_input(casePerEpoch, std::vector<double>(6, 0));
	matrix<double> trainMatrixs_output(casePerEpoch, std::vector<double>(1, 0));
	// [ [num1], [num2], [+], [-], [*], [/] ]
	// [ [ans] ]

	std::random_device rd;
	std::mt19937 gen(rd());

	std::uniform_real_distribution<double> dist(-limit, limit);
	std::uniform_int_distribution<int> opDist(0, 3);



	std::print("正在评估初始 loss...\n");
	std::cout << "\033[?25l";


	double loss = evaluateLoss(casePerEpoch);

	// 显示光标
	std::cout << "\033[?25h";

	std::print("\r样本量: {}, loss = {}\n - 输入形如 1+1 的字符串开始计算\n - 输入 q 开始训练\n", casePerEpoch, loss);
	while (1)
	{
		std::print("input: ");
		std::string input;
		std::getline(std::cin, input);  // 读取整行
		if (input == "q")
		{
			break;
		}
		std::vector<double> inputVec = stringToVector(input);


		std::vector<double> neuroAnsAfter = inverseLogNormalize(NeuroCalc(vectorNorm(inputVec), weightMatrixs, biasMatrixs));

		double neuroRes = neuroAnsAfter[0];
		double rightAns = clacVector(inputVec);


		double delta = abs((rightAns - neuroRes) / rightAns) * 100;

		std::print("\033[1A\r{} {} {} = {} | {} {:.2f}%\n", inputVec[0], vectorToOp(inputVec), inputVec[1], neuroRes, rightAns, delta);



	}











	std::print("\n开始训练\n");

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
			// std::print("\r进度: epoch: {: >{}} / {}, case: {: >{}} / {}\t\t", i + 1, epochDigits, epoch, j + 1, casePerEpochDigits, casePerEpoch);
		}


		std::print("\r进度: epoch: {: >{}} / {}", i + 1, epochDigits, epoch);
	}



	std::print("\n完成!\n");

	double lossAfter = evaluateLoss(casePerEpoch);

	std::print("\r样本量: {}, loss = {}\n", casePerEpoch * epoch, lossAfter);

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






