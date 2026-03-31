#include <iostream>
#include <print>
#include <vector>
#include <stdexcept>


// 矩阵乘法
std::vector<std::vector<double>> matrixMultiply(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b)
{


	int widthA = a[0].size(), heightA = a.size(), widthB = b[0].size(), heightB = b.size();
	std::vector<std::vector<double>> res(heightA, std::vector<double>(widthB, 0.0));
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


int main()
{

}


