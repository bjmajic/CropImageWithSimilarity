//=================================== add by Wang Lei, Version:1.0.0 ===============================
#ifndef SK_IMAGE_CROP_H_
#define SK_IMAGE_CROP_H_

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <iostream>
#include <vector>
#include <cassert>

namespace SK
{
	bool FindSimilarityTransform(const std::vector<Eigen::Matrix<float, 2, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 2, 1>>>& from_points, 
		const std::vector<Eigen::Matrix<float, 2, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 2, 1>>>& to_points, 
		Eigen::Matrix2f& m, Eigen::Vector2f& b);
	Eigen::Vector2f GetAffined2DPoint(const Eigen::Matrix2f& m, const Eigen::Vector2f& b, const Eigen::Vector2f& p);

	bool GetCropedImage( unsigned char* inImgData, int in_w, int in_h, float* basePoint, int basePointNum, 
		float* realPoint, unsigned char* outImgData, int out_w, int out_h, int nMethod = 0);

	/*
	@funtion: 获取对齐后的图片， 这是一个重载函数，基准点固定为
	[28.52+80, 44.70+72],
	[67.29+80, 44.50+72],
	[47.03+80, 64.74+72],
	[32.10+80, 85.37+72],
	[64.20+80, 85.20+72]
			输出图像固定为256*256
	@param:
			inImgData: 输入图像数据，待对齐的图像
			int_w, int_h: 输入图像的宽和高
			realPoint： 输入图像的关键点（对齐的依据）
			outImgData：对齐后的图像
			nMethod： 插值方式  0--最近邻  1--双线性
	*/
	bool GetCropedImage(unsigned char* inImgData, int in_w, int in_h, float* realPoint, unsigned char* outImgData, int nMethod = 0);

	void AffineImageWithBiLinear(unsigned char* srcImage, int srcW, int srcH, unsigned char* destImage, int desW, int desH, const Eigen::Matrix3f& invMat);
	void AffineImageWithNN(unsigned char* srcImage, int srcW, int srcH, unsigned char* destImage, int desW, int desH, const Eigen::Matrix3f& invMat);
} // namespace SK

#endif //SK_IMAGE_CROP_H_