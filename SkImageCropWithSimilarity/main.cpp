#include <iostream>
#include "imagecrop.h"
#include "SKFaceDetectorDLL.h"
using namespace std;

void fun1(string filename, int threadid)
{
	Eigen::Vector2f to_point1;
	vector<Eigen::Matrix<float, 2, 1>, Eigen::aligned_allocator<Eigen::Vector2f>> to_points;
	to_point1 << 28.52f, 44.70f;
	to_points.push_back(to_point1);

	to_point1 << 67.29f, 44.50f;
	to_points.push_back(to_point1);

	to_point1 << 47.03f, 64.74f;
	to_points.push_back(to_point1);

	to_point1 << 32.10f, 85.37f;
	to_points.push_back(to_point1);

	to_point1 << 64.20f, 85.20f;
	to_points.push_back(to_point1);

	float threshold[3] = { 0.6f, 0.7f, 0.72f };
	float factor = 0.709f;
	int minSize = 40;
	std::vector<FaceInfo> *faceInfo = nullptr;
	int w = 0, h = 0;
	unsigned char *bmp = nullptr;

	if (LoadJPG(filename.c_str(), bmp, w, h))
	{

		faceInfo = Detect(bmp, w, h, minSize, threshold, factor);
		if (faceInfo->size() > 0)
		{
			FacePts facePts = (*faceInfo)[0].facePts;
			//Eigen::Vector2f from_point1;
			//vector<Eigen::Matrix<float, 2, 1>, Eigen::aligned_allocator<Eigen::Vector2f>> from_points;
			float realPoints[10];
			float basePoints[10];

			for (int j = 0; j < 5; j++)
			{
				//from_point1 << facePts.y[j], facePts.x[j];
				//from_points.push_back(from_point1);
				realPoints[j * 2] = facePts.y[j];
				realPoints[j * 2 + 1] = facePts.x[j];

				basePoints[j * 2] = to_points[j](0);
				basePoints[j * 2 + 1] = to_points[j](1);
			}

			int rowNumber = 112;
			int colNumber = 96;
			unsigned char *desImage = new unsigned char[rowNumber * colNumber * 3];
			memset(desImage, 0, rowNumber * colNumber * 3);

			SK::GetCropedImage(bmp, w, h, basePoints, 5, realPoints, desImage, colNumber, rowNumber, 0);
			std::string saveFileName = "nn_" + std::to_string(threadid) + ".jpg";
			SaveBGR24JPG(desImage, colNumber, rowNumber, saveFileName.c_str(), 100);

			SK::GetCropedImage(bmp, w, h, basePoints, 5, realPoints, desImage, colNumber, rowNumber, 1);
			std::string saveFileName2 = "bilinear_" + std::to_string(threadid) + ".jpg";
			SaveBGR24JPG(desImage, colNumber, rowNumber, saveFileName2.c_str(), 100);

			delete[] desImage;
			SDKfree(bmp);
		}
	}

	cout << "thread id = " << threadid << " , has end" << endl;
}

void fun2(string filename, int threadid)
{
	float threshold[3] = { 0.6f, 0.7f, 0.72f };
	float factor = 0.709f;
	int minSize = 40;
	std::vector<FaceInfo> *faceInfo = nullptr;
	int w = 0, h = 0;
	unsigned char *bmp = nullptr;

	if (LoadJPG(filename.c_str(), bmp, w, h))
	{

		faceInfo = Detect(bmp, w, h, minSize, threshold, factor);
		if (faceInfo->size() > 0)
		{
			FacePts facePts = (*faceInfo)[0].facePts;
			//Eigen::Vector2f from_point1;
			//vector<Eigen::Matrix<float, 2, 1>, Eigen::aligned_allocator<Eigen::Vector2f>> from_points;
			float realPoints[10];
			

			for (int j = 0; j < 5; j++)
			{
				//from_point1 << facePts.y[j], facePts.x[j];
				//from_points.push_back(from_point1);
				realPoints[j * 2] = facePts.y[j];
				realPoints[j * 2 + 1] = facePts.x[j];

			}

			int rowNumber = 256;
			int colNumber = 256;
			unsigned char *desImage = new unsigned char[rowNumber * colNumber * 3];
			memset(desImage, 0, rowNumber * colNumber * 3);

			SK::GetCropedImage(bmp, w, h, realPoints, desImage, 0);
			std::string saveFileName = "nn_" + std::to_string(threadid) + ".jpg";
			SaveBGR24JPG(desImage, colNumber, rowNumber, saveFileName.c_str(), 100);

			SK::GetCropedImage(bmp, w, h, realPoints, desImage, 1);
			std::string saveFileName2 = "bilinear_" + std::to_string(threadid) + ".jpg";
			SaveBGR24JPG(desImage, colNumber, rowNumber, saveFileName2.c_str(), 100);

			delete[] desImage;
			SDKfree(bmp);
		}
	}

	cout << "thread id = " << threadid << " , has end" << endl;
}
void main()
{
	string imgpath = "F:\\Facescrub_group\\AaronEckhart\\AaronEckhart_2_2.jpg";
	string imgpath2 = "F:\\Facescrub_group\\AaronEckhart\\AaronEckhart_3_3.jpg";

	fun1(imgpath, 1);
	fun1(imgpath2, 2);
	fun2(imgpath, 3);
	fun2(imgpath2, 4);
	system("pause");
}