#include <opencv2/core/core.hpp>  
#include <opencv2/core/mat.hpp>  
#include "cv.h"  
#include "highgui.h"  
#include "cxcore.h"  
#include <iostream>  
#include <fstream>
#include <stdio.h>  
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>    
#include <algorithm>
#include <cmath>
#include <vector>
using namespace cv;
using namespace std;

//对矩阵求和
double sumMat(Mat& inputImg)
{
	double sum = 0.0;
	int rowNumber = inputImg.rows;
	int colNumber = inputImg.cols * inputImg.channels();
	for (int i = 0; i < rowNumber; i++)
	{
		uchar* data = inputImg.ptr<uchar>(i);
		for (int j = 0; j < colNumber; j++)
		{
			sum = data[j] + sum;
		}
	}
	return sum;
}
//对向量求和
float sumvec(Mat& inputImg)
{
	float sum = 0.0;
	for (int i = 0; i <256; i++)
	{
		sum = *(float*)(inputImg.ptr<float>(i)+0) + sum;
	}
	return sum;
}
int main(int argc, char* argv[])
{
	ifstream ifile;
	ifile.open("data.txt");
	float P1[256][80];
	for (int j = 0; j<256; j++)
	{
		for (int i = 0; i<79; i++)
		{
			string value;
			getline(ifile, value, '\t');
			float type = atof(value.c_str());
			P1[j][i] = type;
		}
		int i = 79;
		string value1;
		getline(ifile, value1, '\n');
		float type = atof(value1.c_str());
		P1[j][i] = type;
	}
	ifile.close();
	Mat P(256, 80, CV_32FC1, P1);
	// cout<<P;
	//读取图片，分别是背景图片和前景图片，并转换成CV_32FC1
	vector <Mat> channels;
	vector <Mat> channel1s;
	Mat background1 = imread("../pictures/smoke1667.png");
	Mat foreground1 = imread(argv[1]);//原始图片
	Mat background, foreground;
	cvtColor(background1, background, CV_BGR2GRAY);
	cvtColor(foreground1, foreground, CV_BGR2GRAY);//转成灰度图
	split(background1, channels);
	// background = channels[0];
	channels[2].copyTo(background);
	split(foreground1, channel1s);
	// foreground = channel1s[0];
	channel1s[2].copyTo(foreground);
//  cv::Mat background(480, 640, CV_8UC1);
// 	cv::Mat foreground(480, 640, CV_8UC1);
// 	for(int i=0;i<background1.rows;i++)
//     {
//         for(int j=0;j<background1.cols;j++)
//         {
//             background.at(i, j)=(background1.at(i,j))[2];
//             background.at(i, j)=(foreground1.at(i,j))[2];
//         }
//     }

	cv::Mat b(480, 640, CV_32FC1);
	background.convertTo(b, CV_32FC1);
	cv::Mat f(480, 640, CV_32FC1);
	foreground.convertTo(f, CV_32FC1);//转成CV_32FC1
	int block = 16;
	int m = b.rows;//m是480是行，n是640是列
	int n = b.cols;
	int blockSize = block*block;
	vector<float> alpha, s;

	cv::Mat f3(256, 1, CV_32FC1);
	cv::Mat b3(256, 1, CV_32FC1);
	cv::Mat f31(256, 1, CV_32FC1);
	cv::Mat b31(256, 1, CV_32FC1);
	cv::Mat ones(256, 1, CV_32FC1);
	cv::Mat zarb(256, 1, CV_32FC1);
	cv::Mat f3f3(256, 1, CV_32FC1);
	cv::Mat b3b3(256, 1, CV_32FC1);

	for (int i = 0; i < n; i = i + block)//i是列是x，j是行是y   之前是n
	{
		for (int j = 0; j<m; j = j + block)//i是列是x，j是行是y   之前是m
		{
			Mat src = f(Range(j, j + block), Range(i, i + block));//前面是行，后面是列
			Mat drc = b(Range(j, j + block), Range(i, i + block));//j~j+block-1

			Mat f1, b1;
			src.copyTo(f1);
			drc.copyTo(b1);//
			//std::cout << "f1.type() = " << f1.type() << std::endl;
			f31 = f1.reshape(0, 256);//256*1
			b31 = b1.reshape(0, 256);//256*1
			//cout<<f31<<endl;

			double f3mean = double(sumvec(f31) / 256);
			double b3mean = double(sumvec(b31) / 256);
			//cout << "f3mean" << f3mean << "b3mean" << b3mean << endl;
			ones = Mat::ones(cv::Size(1, 256), CV_32FC1);//256*1,CV_32FC1
			f3 = f31 - f3mean*ones;
			b3 = b31 - b3mean*ones;//256*1

			Mat zarb = f3.mul(b3);//256*1
			double zar = sumvec(zarb);
			f3f3 = f3.mul(f3);
			double zar1 = sumvec(f3f3);
			b3b3 = b3.mul(b3);
			double zar2 = sumvec(b3b3);

			double corr1 = zar / sqrt(zar1*zar2);
			//cout << "a" << endl;
			//cout << corr1 << endl;
			if (corr1<0.001)
			{
				Mat s = (f1 + b1) / 2;
				double alp = 0.5;
				Mat f2 = f31.t();
				Mat b2 = b31.t();//1*256
				for (int h = 0; h<200; h++)
				{
					// +(0.1*Mat::eye(80,80, CV_32FC1))
					Mat first = P*(alp*(P.t()*P) + (0.1*Mat::eye(80, 80, CV_32FC1))).inv();//256*80
					Mat second(80, 1, CV_32FC1);
					second = P.t()*(f2 - b2 + alp*b2).t();
					Mat sp = first*second;//256*1
					//cout << sp << endl;
					Mat third1 = (b2 - sp.t())*(f2 - b2).t();
					double third = double(*(float*)(third1.ptr<float>(0) + 0));
					Mat forth1 = (b2 - sp.t())*(sp.t() - b2).t();
					double forth = double(*(float*)(forth1.ptr<float>(0) + 0));
					double alpstar = third / forth;
					//cout << "third=" << third << "forth=" << forth << endl;
					if (alpstar <= 0)
						alp = 0;
					else if (alpstar >= 1)
						alp = 1;
					else
						alp = alpstar;
				}
				//cout << alp << "  " << endl;
				if (alp>0.3 && alp <= 0.5)
				{
					circle(f, Point(i, j), 5, Scalar(255, 0, 0));
				}
				else if (alp>0.5 && alp <= 0.9)
				{
					circle(f, Point(i, j), 5, Scalar(0, 0, 255));
				}
				else if (alp>0.9)
				{
					circle(f, Point(i, j), 5, Scalar(0, 255, 0));
				}
			}
		}
		cout << "i=" << i << endl;
	}
	cv::Mat fff(480, 640, CV_8UC1);
	f.convertTo(fff, CV_8UC1);
	imshow("src", fff);
	waitKey(0);
	return 0;
}