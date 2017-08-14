#include "imagecrop.h"
#include <cmath>
#include <algorithm>
using namespace std;

namespace SK
{
	bool FindSimilarityTransform(const vector<Eigen::Matrix<float, 2, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 2, 1>>>& from_points,
		const vector<Eigen::Matrix<float, 2, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 2, 1>>>& to_points,
		Eigen::Matrix2f& m, Eigen::Vector2f& b)
	{
		if (from_points.size() != to_points.size() || from_points.size() < 2)
		{
			return false;
		}
		Eigen::Vector2f mean_from;
		Eigen::Vector2f mean_to;
		float sigma_from = 0;
		float sigma_to = 0;
		Eigen::Matrix2f cov;
		cov.fill(0);
		mean_from.fill(0);
		mean_to.fill(0);

		for (int i = 0; i < from_points.size(); ++i)
		{
			mean_from += from_points[i];
			mean_to += to_points[i];
		}
		mean_from /= from_points.size();
		mean_to /= from_points.size();

		for (int i = 0; i < from_points.size(); ++i)
		{
			sigma_from += (from_points[i] - mean_from).squaredNorm();
			sigma_to += (to_points[i] - mean_to).squaredNorm();
			cov += (to_points[i] - mean_to) * ((from_points[i] - mean_from).transpose());
		}
		sigma_from /= from_points.size();
		sigma_to /= from_points.size();
		cov /= from_points.size();

		Eigen::Matrix2f u, v, s, d;
		Eigen::JacobiSVD<Eigen::MatrixXf> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
		u = svd.matrixU();
		v = svd.matrixV();
		d = svd.singularValues().asDiagonal();

		s = Eigen::MatrixXf::Identity(cov.rows(), cov.cols());
		if (cov.determinant() < 0 || (cov.determinant() == 0 && u.determinant() * v.determinant() < 0))
		{
			if (d(1, 1) < d(0, 0))
			{
				s(1, 1) = -1;
			}
			else
			{
				s(0, 0) = -1;
			}
		}
		Eigen::Matrix2f r = u*s*v.transpose();
		float c = 1;
		if (sigma_from != 0)
		{
			c = 1.0f / sigma_from * (d*s).trace();
		}
		Eigen::Vector2f t = mean_to - c*r*mean_from;

		m = c * r;
		b = t;
		return true;
	}

	Eigen::Vector2f GetAffined2DPoint(const Eigen::Matrix2f& m, const Eigen::Vector2f& b, const Eigen::Vector2f& p)
	{
		return m * p + b;
	}

	bool GetCropedImage(unsigned char* inImgData, int in_w, int in_h, float* basePoint, int basePointNum, float* realPoint, unsigned char* outImgData, int out_w, int out_h, int nMethod /* = 0 */)
	{
		
		Eigen::Vector2f from_point;
		vector<Eigen::Matrix<float, 2, 1>, Eigen::aligned_allocator<Eigen::Vector2f>> from_points;
		for (int i = 0; i < basePointNum ; i++)
		{
			from_point << realPoint[(i << 1)], realPoint[(i << 1) + 1];
			from_points.push_back(from_point);
		}

		Eigen::Vector2f to_point;
		vector<Eigen::Matrix<float, 2, 1>, Eigen::aligned_allocator<Eigen::Vector2f>> to_points;
		{
			for (int i = 0; i < basePointNum; i++)
			{
				to_point << basePoint[(i << 1)], basePoint[(i << 1) + 1];
				//to_point.y = basePoint[(i << 1) + 1];
				to_points.push_back(to_point);
			}
			
		}


		Eigen::Matrix2f m;
		Eigen::Vector2f b;

		SK::FindSimilarityTransform(from_points, to_points, m, b);

		Eigen::Matrix3f totalMatrix;
		totalMatrix.fill(0);
		totalMatrix.block<2, 2>(0, 0) = m;
		totalMatrix.block<2, 1>(0, 2) = b;
		totalMatrix(2, 2) = 1;
		Eigen::Matrix3f invMat = totalMatrix.inverse();


		if (0 == nMethod)
		{
			AffineImageWithNN(inImgData, in_w, in_h, outImgData, out_w, out_h, invMat);
		}
		else if (1 == nMethod)
		{
			AffineImageWithBiLinear(inImgData, in_w, in_h, outImgData, out_w, out_h, invMat);
		}
		return true;
	}

	bool GetCropedImage(unsigned char* inImgData, int in_w, int in_h, float* realPoint, unsigned char* outImgData, int sizeMode /*= 2*/, int nMethod /*= 0*/)
	{
		int out_h = 0;
		int out_w = 0;
		Eigen::Vector2f to_point;
		vector<Eigen::Matrix<float, 2, 1>, Eigen::aligned_allocator<Eigen::Vector2f>> to_points;
		switch (sizeMode)
		{
		case 1:
			out_w = 96;
			out_h = 112;
			to_point << 28.52f, 44.70f;
			to_points.push_back(to_point);

			to_point << 67.29f, 44.50f;
			to_points.push_back(to_point);

			to_point << 47.03f, 64.74f;
			to_points.push_back(to_point);

			to_point << 32.10f, 85.37f;
			to_points.push_back(to_point);

			to_point << 64.20f, 85.20f;
			to_points.push_back(to_point);
			break;
		case 2:
			out_w = 256;
			out_h = 256;
			to_point << 108.52f, 116.70f;
			to_points.push_back(to_point);

			to_point << 147.29f, 116.50f;
			to_points.push_back(to_point);

			to_point << 127.03f, 136.74f;
			to_points.push_back(to_point);

			to_point << 112.10f, 157.37f;
			to_points.push_back(to_point);

			to_point << 144.20f, 157.20f;
			to_points.push_back(to_point);
			break;
		default:
			out_w = 256;
			out_h = 256;
			to_point << 108.52f, 116.70f;
			to_points.push_back(to_point);

			to_point << 147.29f, 116.50f;
			to_points.push_back(to_point);

			to_point << 127.03f, 136.74f;
			to_points.push_back(to_point);

			to_point << 112.10f, 157.37f;
			to_points.push_back(to_point);

			to_point << 144.20f, 157.20f;
			to_points.push_back(to_point);
			break;
		}

		Eigen::Vector2f from_point;
		vector<Eigen::Matrix<float, 2, 1>, Eigen::aligned_allocator<Eigen::Vector2f>> from_points;
		for (int i = 0; i < 5; i++)
		{
			from_point << realPoint[(i << 1)], realPoint[(i << 1) + 1];
			from_points.push_back(from_point);
		}
				
		Eigen::Matrix2f m;
		Eigen::Vector2f b;

		SK::FindSimilarityTransform(from_points, to_points, m, b);

		Eigen::Matrix3f totalMatrix;
		totalMatrix.fill(0);
		totalMatrix.block<2, 2>(0, 0) = m;
		totalMatrix.block<2, 1>(0, 2) = b;
		totalMatrix(2, 2) = 1;
		Eigen::Matrix3f invMat = totalMatrix.inverse();

		if (0 == nMethod)
		{
			AffineImageWithNN(inImgData, in_w, in_h, outImgData, out_w, out_h, invMat);
		}
		else if (1 == nMethod)
		{
			AffineImageWithBiLinear(inImgData, in_w, in_h, outImgData, out_w, out_h, invMat);
		}
		return true;
	}

	void AffineImageWithNN(unsigned char* srcImage, int srcW, int srcH, unsigned char* destImage, int desW, int desH, const Eigen::Matrix3f& invMat)
	{
		int rowNumber = desH;
		int colNumber = desW;
		//unsigned char *desImage = new unsigned char[rowNumber * colNumber * 3];
		memset(destImage, 0, rowNumber * colNumber * 3);
		int linesize = colNumber * 3;
		unsigned char *pdesImage = destImage + (rowNumber - 1) * linesize;
		linesize = linesize << 1;
		int h_1 = srcH - 1;
		for (int i = 0; i < rowNumber; ++i)
		{
			for (int j = 0; j < colNumber; ++j)
			{
				Eigen::Vector3f p;
				p.fill(0);
				Eigen::Vector3f result_p;
				result_p.fill(0);

				//p.x = j;
				//p.y = i;
				//p.z = 1;
				p << j, i, 1;

				result_p = invMat * p;
				int rowIndex = std::round(result_p(1));
				int colIndex = std::round(result_p(0));

				if (rowIndex >= 0 && colIndex >= 0 && rowIndex < srcH && colIndex < srcW)
				{
					int ptr = ((h_1 - rowIndex) * srcW + colIndex) * 3;
					*(pdesImage++) = srcImage[ptr++];
					*(pdesImage++) = srcImage[ptr++];
					*(pdesImage++) = srcImage[ptr];
				}
				else
				{
					pdesImage += 3;
				}
			}
			pdesImage -= linesize;
		}
	}

	void AffineImageWithBiLinear(unsigned char* srcImage, int srcW, int srcH, unsigned char* destImage, int desW, int desH, const Eigen::Matrix3f& invMat)
	{
		int rowNumber = desH;
		int colNumber = desW;
		//unsigned char *desImage = new unsigned char[rowNumber * colNumber * 3];
		memset(destImage, 0, rowNumber * colNumber * 3);

		for (int i = 0; i < rowNumber; ++i)
		{
			for (int j = 0; j < colNumber; ++j)
			{
				Eigen::Vector3f p;
				p.fill(0);
				Eigen::Vector3f result_p;
				result_p.fill(0);

				p << j, i, 1; //ÒÔx, yµÄË³Ðò¸³Öµ
				result_p = invMat * p;

				int rowIndex1 = static_cast<int>(std::floor((result_p(1))));
				int colIndex1 = static_cast<int>(std::floor((result_p(0))));

				int rowIndex2 = static_cast<int>(std::ceil((result_p(1))));
				int colIndex2 = static_cast<int>(std::ceil((result_p(0))));
				if (rowIndex1 >= 0 && colIndex1 >= 0 && rowIndex1 < srcH && colIndex1 < srcW &&
					rowIndex2 >= 0 && colIndex2 >= 0 && rowIndex2 < srcH && colIndex2 < srcW)
				{
					unsigned char b11 = *(srcImage + (srcH - 1 - rowIndex1) * srcW * 3 + colIndex1 * 3 + 0);
					unsigned char g11 = *(srcImage + (srcH - 1 - rowIndex1) * srcW * 3 + colIndex1 * 3 + 1);
					unsigned char r11 = *(srcImage + (srcH - 1 - rowIndex1) * srcW * 3 + colIndex1 * 3 + 2);

					unsigned char b21 = *(srcImage + (srcH - 1 - rowIndex1) * srcW * 3 + colIndex2 * 3 + 0);
					unsigned char g21 = *(srcImage + (srcH - 1 - rowIndex1) * srcW * 3 + colIndex2 * 3 + 1);
					unsigned char r21 = *(srcImage + (srcH - 1 - rowIndex1) * srcW * 3 + colIndex2 * 3 + 2);

					unsigned char b12 = *(srcImage + (srcH - 1 - rowIndex2) * srcW * 3 + colIndex1 * 3 + 0);
					unsigned char g12 = *(srcImage + (srcH - 1 - rowIndex2) * srcW * 3 + colIndex1 * 3 + 1);
					unsigned char r12 = *(srcImage + (srcH - 1 - rowIndex2) * srcW * 3 + colIndex1 * 3 + 2);

					unsigned char b22 = *(srcImage + (srcH - 1 - rowIndex2) * srcW * 3 + colIndex2 * 3 + 0);
					unsigned char g22 = *(srcImage + (srcH - 1 - rowIndex2) * srcW * 3 + colIndex2 * 3 + 1);
					unsigned char r22 = *(srcImage + (srcH - 1 - rowIndex2) * srcW * 3 + colIndex2 * 3 + 2);


					float factorx1 = 0;//= (result_p(0) - rowIndex1) / (rowIndex2 - rowIndex1);
					float factorx2 = 0;//= (rowIndex2 - result_p(1)) / (rowIndex2 - rowIndex1);
					float factory1 = 0;//= (result_p(1) - colIndex1) / (colIndex2 - colIndex1);
					float factory2 = 0;//= (colIndex2 - result_p(1)) / (colIndex2 - colIndex1);

					if (colIndex1 == colIndex2)
					{
						factorx1 = 0.5;
						factorx2 = 0.5;
					}
					else
					{
						factorx1 = (result_p(0) - colIndex1) / (colIndex2 - colIndex1);
						factorx2 = (colIndex2 - result_p(0)) / (colIndex2 - colIndex1);
					}

					if (rowIndex1 == rowIndex2)
					{
						factory1 = 0.5;
						factory2 = 0.5;
					}
					else
					{
						factory1 = (result_p(1) - rowIndex1) / (rowIndex2 - rowIndex1);
						factory2 = (rowIndex2 - result_p(1)) / (rowIndex2 - rowIndex1);
					}

					float direct_x_value_b1 = factorx2 * b11 + factorx1 * b21;
					float direct_x_value_g1 = factorx2 * g11 + factorx1 * g21;
					float direct_x_value_r1 = factorx2 * r11 + factorx1 * r21;

					float direct_x_value_b2 = factorx2 * b12 + factorx1 * b22;
					float direct_x_value_g2 = factorx2 * g12 + factorx1 * g22;
					float direct_x_value_r2 = factorx2 * r12 + factorx1 * r22;

					float res_b = factory2 * direct_x_value_b1 + factory1 * direct_x_value_b2;
					float res_g = factory2 * direct_x_value_g1 + factory1 * direct_x_value_g2;
					float res_r = factory2 * direct_x_value_r1 + factory1 * direct_x_value_r2;

					*(destImage + (rowNumber - 1 - i)*colNumber * 3 + j * 3 + 0) = static_cast<unsigned char>(res_b);
					*(destImage + (rowNumber - 1 - i)*colNumber * 3 + j * 3 + 1) = static_cast<unsigned char>(res_g);
					*(destImage + (rowNumber - 1 - i)*colNumber * 3 + j * 3 + 2) = static_cast<unsigned char>(res_r);
				}
			}
		}
	}
} // namespace SK