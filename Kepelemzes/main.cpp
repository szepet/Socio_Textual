//BEAD1
/*
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <ctime>
#include <vector>

cv::Mat orig, newpic1, newpic2;
int N = 5;

cv::Vec3b avgs1(int x, int y) {
	cv::Vec3i sums(0, 0, 0);
	for (int i = x - N; i <= x + N; ++i)
		for (int j = y - N; j <= y + N; ++j)
			sums += orig.at<cv::Vec3b>(j, i);

	return sums / ((2 * N + 1)*(2 * N + 1));
}

int main() {
	N = 28;
	orig = cv::imread("lena.png");
	newpic1 = cv::imread("lena.png");
	newpic2 = cv::imread("lena.png");

	std::clock_t start1 = std::clock();
	for (int i = N; i < orig.cols - N; i++)
		for (int j = N; j < orig.rows - N; j++)
		{
			newpic1.at<cv::Vec3b>(j, i) = avgs1(i, j);
		}
	std::clock_t end1 = std::clock();

	std::cout << "Normal szuro futasi ideje: " << end1 - start1 << " ms\n";

	cv::imwrite("outpic_1.png", newpic1);

	std::clock_t start2 = std::clock();

	std::vector<cv::Vec3i> S(orig.cols);
	for (int i = 0; i < S.size(); i++)
		S[i] = cv::Vec3i(0, 0, 0);

	cv::Vec3i Sw(0, 0, 0);
	// S init
	for (int i = 0; i < orig.cols; i++)
		for (int j = 0; j < 2 * N + 1; j++)
			S[i] += orig.at<cv::Vec3b>(i, j);

	for (int i = N; i < orig.rows - N; i++) {
		// S update
		for (int k = 0; k < S.size() && i != N; k++) {
			S[k] -= orig.at<cv::Vec3b>(k, i - N - 1);
			S[k] += orig.at<cv::Vec3b>(k, i + N);
		}

		// Sw init
		Sw = 0;
		for (int j = 0; j < 2 * N + 1; j++)
			Sw += S[j];

		for (int j = N; j < orig.cols - N; j++)
		{
			// Sw update
			if (j != N) {
				Sw -= S[j - N - 1];
				Sw += S[j + N];
			}

			newpic2.at<cv::Vec3b>(j, i) = Sw / ((2 * N + 1)*(2 * N + 1));
		}
	}
	std::clock_t end2 = std::clock();
	std::cout << "Gyors futo szuro futasi ideje: " << end2 - start2 << " ms\n";

	cv::imwrite("outpic_2.png", newpic2);
}



*/
//BEAD2
/*
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <ctime>
#include <vector>
#include <fstream>
#include <cmath>
cv::Mat orig, newpic1, newpic2;

void dude(int x, int y, float& arc, float& length) {
	float fx = 0, fy = 0;
	for (int i = x - 1; i <= x + 1; ++i) {
		for (int j = y - 1; j <= y + 1; ++j) {
			fy += (1.0f / 3.0f)*(y - j)*orig.at<uchar>(j, i);
			fx += (1.0f / 3.0f)*(x - i)*orig.at<uchar>(j, i);
		}
	}
	length = sqrt(fx*fx + fy*fy);
	arc = atanf(fx/fy);
}

#define M_PI 3.14159265358979323846f
int main() {
	const std::string filename = "lena.png";
	orig = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	std::ofstream os("kifele.txt");
	cv::Mat arcs = cv::Mat_<float>(orig.rows, orig.cols);
	cv::Mat zs = cv::Mat_<float>(orig.rows, orig.cols);
	newpic1 = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	newpic2 = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat arcpic = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	//CV_32FC1
	std::clock_t start1 = std::clock();
	for (int i = 1; i < orig.cols - 1; i++) {
		for (int j = 1; j < orig.rows - 1; j++)
		{
			float z, arc;
			dude(i, j, arc, z);
			arcs.at<float>(j, i) = arc;
			arcpic.at<uchar>(j, i) = arc;
			newpic1.at<uchar>(j,i) = z;
			zs.at<float>(j, i) = z;
			os << "(" << z << "," << arc << ") ";
		}
		os << "\n";
	}

	for (int i = 0; i < orig.cols; i++)
	{
		zs.at<float>(0,i) = 255;
		arcs.at<float>(0,i) = 0;
		zs.at<float>(orig.rows-1, i) = 255;
		arcs.at<float>(orig.rows-1, i) = 0;
	}

	for (int i = 0; i < orig.rows; i++)
	{
		zs.at<float>(i, 0) = 255;
		arcs.at<float>(i, 0) = M_PI / 2;
		zs.at<float>(i, orig.cols-1) = 255;
		arcs.at<float>(i, orig.cols - 1) = M_PI/2;
	}

	float arc, z;
	for (int i = 1; i < orig.cols - 1; i++) {
		for (int j = 1; j < orig.rows - 1; j++)
		{
			z = zs.at<float>(j, i);
			arc = arcs.at<float>(j, i);
			while (arc > M_PI)
				arc -= M_PI;
			while (arc < 0)
				arc += M_PI;

			float m1 = 0, m2 = 0;
			if (arc < M_PI / 8 || arc >= M_PI*7.0 / 8.0) {
				m1 = zs.at<float>(j - 1, i);
				m2 = zs.at<float>(j + 1, i);

			}
			if (arc >= M_PI / 8 && arc < M_PI*3.0 / 8.0) {
				m1 = zs.at<float>(j - 1, i - 1);
				m2 = zs.at<float>(j + 1, i + 1);

			}
			if (arc >= M_PI*3.0 / 8.0 && arc < M_PI*5.0 / 8.0) {
				m1 = zs.at<float>(j, i - 1);
				m2 = zs.at<float>(j, i + 1);
			}
			if (arc >= M_PI*5.0 / 8.0 && arc < M_PI*7.0 / 8.0) {
				m1 = zs.at<float>(j - 1, i + 1);
				m2 = zs.at<float>(j + 1, i - 1);
			}
			if (m1 > z || m2 >= z || z < 10)
				newpic2.at<uchar>(j,i) = 0;
			else
				newpic2.at<uchar>(j, i) = 255;
		}
	}


	std::clock_t end1 = std::clock();

	cv::Mat harom_kep_egyutt(cv::Size(3 * newpic2.cols, newpic2.rows), CV_8UC1, cv::Scalar(0));
	orig.copyTo(harom_kep_egyutt.colRange(0, newpic2.cols));
	newpic1.copyTo(harom_kep_egyutt.colRange(newpic2.cols, newpic2.cols*2));
	newpic2.copyTo(harom_kep_egyutt.colRange(newpic2.cols*2, newpic2.cols*3));

	//cv::imshow("Result", newpic2);
	//cv::imshow("Grad", newpic1);
	//cv::imshow("Orig", orig);
	cv::imshow("FULLPOWER!!!!!!", harom_kep_egyutt);
	cv::waitKey(0);

	cv::imwrite("grad.png", newpic1);
	cv::imwrite("result.png", newpic2);
	//cv::imwrite("arckep.png", arcpic);

}
*/
//BEAD3
/*
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <ctime>
#include <vector>
#include <fstream>
#include <cmath>
#include <tuple>
cv::Mat orig, newpic1, newpic2;
//cv::Mat fxs, fys, fxfys;

void dude(int x, int y, float& _fx, float& _fy) {
	float fx = 0, fy = 0;
	for (int i = x - 1; i <= x + 1; ++i) {
		for (int j = y - 1; j <= y + 1; ++j) {
			fy += (y - j)*newpic1.at<float>(j, i);
			fx += (x - i)*newpic1.at<float>(j, i);
		}
	}
	_fx = fx/3;
	_fy = fy/3;
}


int N, M;

float avgs(int x, int y, cv::Mat& Mt) {
	float sums = 0;
	for (int i = x - N; i <= x + N; ++i)
		for (int j = y - N; j <= y + N; ++j)
			sums += Mt.at<float>(j, i);

	return sums / (M*M);
}


bool haspointnear(int x, int y, cv::Mat& Mt) {
	for (int i = std::max(x - N,0); i <= std::min(x + N,Mt.cols-1); ++i)
		for (int j = std::max(y - N, 0); j <= std::min(y + N, Mt.rows -1); ++j)
			if ((j!=y || x!=i) && Mt.at<uchar>(j, i) > 0)
				return true;
	//std::cout << "DUDE";
	return false;
}



#define M_PI 3.14159265358979323846f
int main() {
	float alpha, thr;
	std::cout << "window_size alpha threshold" << std::endl;
	std::cin >> M >> alpha >> thr;
	//M = 5; alpha = 0.05; thr = 0.000001;
	N = M / 2;

	const std::string filename = "house.png";
	cv::Mat vegsoout = cv::imread(filename);

	orig = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

	std::ofstream os("kifele.txt");

	newpic1 = cv::Mat_<float>(orig.rows, orig.cols);
	cv::Mat fxs = cv::Mat_<float>(orig.rows, orig.cols);
	cv::Mat fys = cv::Mat_<float>(orig.rows, orig.cols);
	cv::Mat fxfys = cv::Mat_<float>(orig.rows, orig.cols);

	cv::Mat fxs2 = cv::Mat_<float>(orig.rows, orig.cols);
	cv::Mat fys2 = cv::Mat_<float>(orig.rows, orig.cols);
	cv::Mat fxfys2 = cv::Mat_<float>(orig.rows, orig.cols);

	cv::Mat H = cv::Mat_<float>(orig.rows, orig.cols);

	newpic2 = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat arcpic = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	//CV_32FC1
	// NORMALIZALAS
	for (int i = 0; i < orig.cols; i++) {
		for (int j = 0; j < orig.rows; j++)
		{
			newpic1.at<float>(j, i) = float(orig.at<uchar>(j, i)) / 255.0f;
		}
	}
	std::cerr << "Dude0\n";
	for (int i = 1; i < orig.cols - 1; i++) {
		for (int j = 1; j < orig.rows - 1; j++)
		{
			float fx, fy;
			dude(i, j, fx, fy);
			fxs.at<float>(j, i) = fx*fx;
			fys.at<float>(j, i) = fy*fy;
			fxfys.at<float>(j, i) = fx*fy;
			//os << fx << " " <<fy << std::endl;
		}
	}
	//
	std::cerr << "Dude1\n";
	for (int i = 0; i < orig.cols; i++)
	{
		fxs.at<float>(0, i) = 0;
		fys.at<float>(0, i) = 0;
		fxfys.at<float>(0, i) = 0;
		fxs.at<float>(orig.rows - 1, i) = 0;
		fys.at<float>(orig.rows - 1, i) = 0;
		fxfys.at<float>(orig.rows - 1, i) = 0;
	}

	for (int i = 0; i < orig.rows; i++)
	{
		fxs.at<float>(i, 0) = 0;
		fys.at<float>(i, 0) = 0;
		fxfys.at<float>(i, 0) = 0;
		fxs.at<float>(i, orig.cols - 1) = 0;
		fys.at<float>(i, orig.cols - 1) = 0;
		fxfys.at<float>(i, orig.cols - 1) = 0;
	}
	std::cerr << "Dude2\n";

// DOBOZOLAS
	for (int i = N; i < orig.cols - N; i++)
		for (int j = N; j < orig.rows - N; j++)
		{
			fxs2.at<float>(j, i) = avgs(i, j, fxs);
			fys2.at<float>(j, i) = avgs(i, j, fys);
			fxfys2.at<float>(j, i) = avgs(i, j, fxfys);
		}
	std::cerr << "Dude3\n";
	std::vector<std::tuple<int, int, float>> v;
	// MEGVAN A DOBOZOLAS
	for (int i = N; i < orig.cols - N; i++)
		for (int j = N; j < orig.rows - N; j++)
		{
			float fx2 = fxs2.at<float>(j, i);
			float fy2 = fys2.at<float>(j, i);
			float fxfy = fxfys2.at<float>(j, i);
			H.at<float>(j, i) = fx2*fy2 - fxfy*fxfy - alpha*(fx2 + fy2)*(fx2 + fy2);
			os << fx2 << " " << fy2 << " " <<fxfy << " "<< H.at<float>(j, i) << std::endl;
			if (H.at<float>(j, i)>thr) {
				newpic2.at<uchar>(j, i) = 255;
				v.push_back(std::make_tuple(i,j, H.at<float>(j, i)));
			}
			else
				newpic2.at<uchar>(j, i) = 0;
			//H.at<float>(j, i) * 255;
		}


	// utofeldolgozas
	cv::Mat newpic3 = newpic2.clone();
	std::sort(v.begin(), v.end(), [](const std::tuple<int,int,float>& t1, const std::tuple<int, int, float>& t2) {
		return std::get<2>(t1) < std::get<2>(t2); });
	for (int i = 0; i < v.size(); i++)
	{
		int k = std::get<0>(v[i]);
		int j = std::get<1>(v[i]);
		//std::cerr << k << " " << j << "\n";
		if(haspointnear(k,j,newpic3))
			newpic3.at<uchar>(j, k) = 0;
	}


	for (int i = N; i < orig.cols - N; i++)
		for (int j = N; j < orig.rows - N; j++)
		{
			if (newpic3.at<uchar>(j, i)>0)
				vegsoout.at<cv::Vec3b>(j, i) = cv::Vec3b(0,0,255);
		}


	cv::Mat harom_kep_egyutt(cv::Size(3 * newpic2.cols, newpic2.rows), CV_8UC1, cv::Scalar(0));
	orig.copyTo(harom_kep_egyutt.colRange(0, newpic2.cols));
	newpic2.copyTo(harom_kep_egyutt.colRange(newpic2.cols, newpic2.cols * 2));
	newpic3.copyTo(harom_kep_egyutt.colRange(newpic2.cols * 2, newpic2.cols * 3));

	//cv::imshow("Result", newpic2);
	//cv::imshow("Grad", newpic1);
	//cv::imshow("Orig", orig);
	//cv::imshow("FULLPOWER!!!!!!", harom_kep_egyutt);
	//cv::waitKey(0);

	cv::imwrite("vegso.png", vegsoout);
	cv::imwrite("newpic2.png", newpic2);
	//cv::imwrite("arckep.png", arcpic);

}
*/
//BEAD4
/*
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <ctime>
#include <vector>
cv::Mat orig, newpic1, newpic2;
int main() {
	const std::string filename = "finger.png";
	orig = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	std::vector<float> hist(256);
	cv::Mat arcs = cv::Mat_<float>(orig.rows, orig.cols);
	cv::Mat zs = cv::Mat_<float>(orig.rows, orig.cols);
	newpic1 = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat arcpic = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	//CV_32FC1
	std::clock_t start1 = std::clock();
	for (int i = 0; i < orig.cols; i++) {
		for (int j = 0; j < orig.rows; j++)
		{
			hist[orig.at<uchar>(j, i)] += 1;
		}
	}


	for (int i = 0; i < 256; i++)
		hist[i] /= orig.rows * orig.cols;
	//std::cerr << "lul" << std::endl;
	//for (int i = 0; i < 256; i++)
	//	std::cout << hist[i] << " ";
	//std::cout << std::endl;

	float sum = 0;
	for (int i = 0; i < 256; i++)
		sum += hist[i];
	//std::cout << sum << " " << orig.rows * orig.cols << std::endl;

	float nu = 0;
	for (int i = 0; i < 256; i++)
		nu += i*hist[i];

	std::vector<float> q(256), nu1(256), nu2(256), sig(256);
	q[0] = hist[0];
	nu1[0] = 0;
	int N = 0;
	while (hist[N] < 0.00000000000001f)
		N++;

	q[N] = hist[N];
	nu1[N] = 0;

	int M = 254;
	while (hist[M] < 0.00000000000001f)
		M--;
	//std::cout << "M: " << M << std::endl;

	for (int t = N; t < M; t++)
		q[t + 1] = q[t] + hist[t + 1];

	for (int t = N; t < M - 1; t++)
		nu1[t + 1] = (q[t] * nu1[t] + (t + 1) * hist[t + 1]) / q[t + 1];

	for (int t = N; t < M - 1; t++)
		nu2[t] = (nu - q[t] * nu1[t]) / (1 - q[t]);

	int topt = -1;
	float toptval = -10000000000.f;

	for (int t = std::max(N, 1); t < std::min(M - 1, 255); t++) {
		sig[t] = q[t] * (1 - q[t])*(nu1[t] - nu2[t])*(nu1[t] - nu2[t]);
		if (sig[t] < 0) std::cout << "Whatdafakk\n";
		//std::cout << nu1[t] << " " << nu2[t] << std::endl;
		if (sig[t] > toptval) {
			topt = t;
			toptval = sig[t];
		}
	}

	for (int i = 0; i < orig.cols; i++) {
		for (int j = 0; j < orig.rows; j++)
		{
			if (orig.at<uchar>(j, i)>topt)
				newpic1.at<uchar>(j, i) = 255;
			else
				newpic1.at<uchar>(j, i) = 0;
		}
	}
	std::cerr << topt << std::endl;

	cv::imshow("Grad", newpic1);
	cv::imshow("Orig", orig);
	//cv::imshow("FULLPOWER!!!!!!", harom_kep_egyutt);
	cv::waitKey(0);

}*/
//BEAD5
/*
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <ctime>
#include <vector>
cv::Mat orig, newpic1, newpic2;


int nz(int x, int y) {
	int cnt = 0;
	for (int i = x - 1; i <= x + 1; ++i)
		for (int j = y - 1; j <= y + 1; ++j)
			if ((j != y || x != i) && newpic1.at<uchar>(j, i) == 255)
				cnt++;
	//std::cout << "DUDE";
	return cnt;
}


int tr(int i, int j) {
	std::vector<int> v(9);
	v[0] = newpic1.at<uchar>(j - 1, i);
	v[1] = newpic1.at<uchar>(j - 1, i - 1);
	v[2] = newpic1.at<uchar>(j, i - 1);
	v[3] = newpic1.at<uchar>(j + 1, i - 1);
	v[4] = newpic1.at<uchar>(j + 1, i);
	v[5] = newpic1.at<uchar>(j + 1, i + 1);
	v[6] = newpic1.at<uchar>(j, i + 1);
	v[7] = newpic1.at<uchar>(j - 1, i + 1);
	v[8] = newpic1.at<uchar>(j - 1, i);
	int cnt = 0;
	for (int i = 0; i < 8; ++i) {
		if (v[i] == 0 && v[i + 1] == 255)
			cnt++;
	}
	return cnt;
}


int main() {
	const std::string filename = "csiga.png";
	orig = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	//newpic1 = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	newpic2 = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

	//newpic2 = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat TR = cv::Mat_<int>(orig.rows, orig.cols, 0);
	cv::Mat NZ = cv::Mat_<int>(orig.rows, orig.cols, 0);

	//CV_32FC1
	std::clock_t start1 = std::clock();
	for (int i = 0; i < orig.cols; i++) {
		for (int j = 0; j < orig.rows; j++)
		{
			orig.at<uchar>(j, i) = 255 - orig.at<uchar>(j, i);
			newpic2.at<uchar>(j, i) = 255 - newpic2.at<uchar>(j, i);
		}
	}



	bool again = true;
	while (again) {
		again = false;
		newpic2.copyTo(newpic1);
		cv::imshow("Grad", newpic2);
		//cv::imshow("FULLPOWER!!!!!!", harom_kep_egyutt);
		cv::waitKey(0);
		for (int i = 1; i < orig.cols - 1; i++) {
			for (int j = 1; j < orig.rows - 1; j++)
			{
				TR.at<int>(j, i) = tr(i, j);
				NZ.at<int>(j, i) = nz(i, j);
			}
		}

		for (int i = 1; i < orig.cols - 1; i++) {
			for (int j = 1; j < orig.rows - 1; j++)
			{
				int p2 = newpic1.at<uchar>(j - 1, i); //p2
				int p4 = newpic1.at<uchar>(j, i - 1); //p4
				int p6 = newpic1.at<uchar>(j + 1, i); //p6
				int p8 = newpic1.at<uchar>(j, i + 1); //p8
				if (newpic1.at<uchar>(j, i) == 255 && NZ.at<int>(j, i) >= 2 && NZ.at<int>(j, i) <= 6 &&
					TR.at<int>(j, i) == 1 && (p2*p4*p8 == 0 || TR.at<int>(j - 1, i) != 1) &&
					(p2*p4*p6 == 0 || TR.at<int>(j, i - 1) != 1)) {
					again = true;
					newpic2.at<uchar>(j, i) = 0;
				}
			}
		}
	}
	for (int i = 0; i < orig.cols; i++) {
		for (int j = 0; j < orig.rows; j++)
		{
			orig.at<uchar>(j, i) = 255 - orig.at<uchar>(j, i);
			newpic2.at<uchar>(j, i) = 255 - newpic2.at<uchar>(j, i);
		}
	}

	cv::imshow("Grad", newpic2);

	cv::waitKey(0);

}*/




#include <iostream>
#include <set>
#include <map>
#include <vector>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <string>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;

typedef std::pair<double, double> Geotag;
typedef int User;

std::ostream& operator<<(std::ostream& os, const Geotag& gt) {
	os << '(' << gt.first << ',' << gt.second << ')';
	return os;
}

std::ostream& operator<<(std::ostream& os, const std::set<Geotag>& gtset) {
	os << '{';
	for (auto& r : gtset)
		os << r;
	os << "}";
	return os;
}

std::ostream& operator<<(std::ostream& os, const std::set<std::set<Geotag>>& gtsetset) {
	os << '[';
	for (auto& r : gtsetset)
		os << r << ",\n";
	os << "]";
	return os;
}

double dist(Geotag g1, Geotag g2) {
	return sqrt((g1.first - g2.first)*(g1.first - g2.first) + (g1.second - g2.second)*(g1.second - g2.second));
}

struct Post {
	Geotag loc;
	User user;
	std::set<std::string> keywords;
	Post(const Geotag& l, const User& u, const std::set<std::string>& kw) :loc(l), user(u), keywords(kw) {}
};

std::set<User> Users;
std::map<User, std::vector<Post>> UserPosts;
std::map<Geotag, Post> LocToPost;
std::set<Geotag> Locations;

//Algorithm 2
//Input: keyword set psi
//Output: set Upsi of relevant users
std::set<int> identifyRelevantUsers(const std::set<std::string>& psi) {
	std::set<User> Upsi;
	for (auto u : Users) {
		std::set<std::string> covpsi;
		for (auto p : UserPosts[u])
			for (auto &keyword : p.keywords)
				if (psi.count(keyword))
					covpsi.insert(keyword);
		if (covpsi.size() == psi.size())
			Upsi.insert(u);
	}
	return Upsi;
}

std::set<User> Upsi;
std::map<std::set<Geotag>, int> rw_sup, sup;

//Algorithm 3: ComputeSupports
//Input: location set L, keyword set Ψ
//Output: weak support and support of(L,Ψ)
void computeSupports(const std::set<Geotag> & L, const std::set<std::string>& psi) {
	for (auto u : Upsi) {
		std::set<Geotag> covL;
		std::set<std::string> covpsi;

		for (auto p : UserPosts[u])
			for (auto l : L)
				if (dist(p.loc, l) < 0.2) {
					std::set<std::string> intersect;
					std::set_intersection(psi.begin(), psi.end(), p.keywords.begin(), p.keywords.end(), std::inserter(intersect, intersect.begin()));
					for (auto& ps : intersect) {
						covL.insert(l);
						covpsi.insert(ps);
					}
				}

		if (covL.size() == L.size()) {
			rw_sup[L]++;
			if (covpsi.size() == psi.size())
				sup[L]++;
		}
	}
}

std::vector<std::set<std::set<Geotag>>> C;
//std::set<std::set<Geotag>> powerSetsWithLength(const std::set<Geotag>& L, int n) { return std::set<std::set<Geotag>>(); }
void combinationUtil(const std::set<std::set<Geotag>>& Fi, int r, int index, Geotag data[], std::set<Geotag>::const_iterator it);
void candidateGeneration(const std::set<std::set<Geotag>>& Fi, int r)
{
	Geotag data[12];
	combinationUtil(Fi, r, 0, data, Locations.begin());
}

void combinationUtil(const std::set<std::set<Geotag>>& Fi, int r, int index, Geotag data[], std::set<Geotag>::const_iterator it)
{
	if (index == r) {
		bool b = true;
		auto newset = std::set<Geotag>();
		for (int j = 0; j < r; j++) {
			newset.insert(data[j]);
		}
		//std::cout << "newset: " << newset;
		for (int j = 0; j < r; j++) {
			newset.erase(data[j]);
			//std::cout << "newsettest: " << newset;
			if (!Fi.count(newset))
				b = false;
			//std::cout << "b: " << b << std::endl;

			newset.insert(data[j]);
		}
		if (b)
			C[r - 1].insert(newset);
		return;
	}

	// When no more elements are there to put in Locations
	if (it == Locations.end())
		return;

	data[index] = *it;
	combinationUtil(Fi, r, index + 1, data, ++it);
	combinationUtil(Fi, r, index, data, it);
}

cv::Vec3b getColor(const std::string& s) {
	return cv::Vec3b(0, 0, 0);
}
const int D = 500;
void markObject(cv::Mat& Pic, const std::set<Geotag> L)
{
	Pic.at<cv::Vec3b>(D*(L.begin()->first), D*(L.begin()->second)) = getColor(*(LocToPost.at(*(L.begin())).keywords.begin()));
	Pic.at<cv::Vec3b>(D*(L.begin()->first) + 1, D*(L.begin()->second)) = getColor(*(LocToPost.at(*(L.begin())).keywords.begin()));
	Pic.at<cv::Vec3b>(D*(L.begin()->first), D*(L.begin()->second) + 1) = getColor(*(LocToPost.at(*(L.begin())).keywords.begin()));
	Pic.at<cv::Vec3b>(D*(L.begin()->first) - 1, D*(L.begin()->second)) = getColor(*(LocToPost.at(*(L.begin())).keywords.begin()));
	Pic.at<cv::Vec3b>(D*(L.begin()->first), D*(L.begin()->second) - 1) = getColor(*(LocToPost.at(*(L.begin())).keywords.begin()));
	Pic.at<cv::Vec3b>(D*(L.begin()->first) + 1, D*(L.begin()->second) + 1) = getColor(*(LocToPost.at(*(L.begin())).keywords.begin()));
	Pic.at<cv::Vec3b>(D*(L.begin()->first) - 1, D*(L.begin()->second) - 1) = getColor(*(LocToPost.at(*(L.begin())).keywords.begin()));
	Pic.at<cv::Vec3b>(D*(L.begin()->first) + 1, D*(L.begin()->second) - 1) = getColor(*(LocToPost.at(*(L.begin())).keywords.begin()));
	Pic.at<cv::Vec3b>(D*(L.begin()->first) - 1, D*(L.begin()->second) + 1) = getColor(*(LocToPost.at(*(L.begin())).keywords.begin()));
}

int main()
{

	cv::Mat Pic = cv::Mat_<cv::Vec3b>(D, D);


	for (int i = 0; i < 500; i++)
		for (int j = 0; j < 500; j++)
		{
			if(abs((-i + 2*j-200)) < 70)
				Pic.at<cv::Vec3b>(j, i) = cv::Vec3b(225, 0, 20);
			else
				Pic.at<cv::Vec3b>(j, i) = cv::Vec3b(10, 125, 70);
		}

	//cv::imshow("Grad", Pic);
	//cv::waitKey(0);
	int N;
	ifstream ifs("data.txt");
	ifs >> N;
	std::cout << "N: " << N << std::endl;
	User uid;
	std::string tmpString, tmpString2;
	double tmplon, tmplat;
	for (int i = 0; i < N; i++)
	{
		ifs >> uid;
		Users.insert(uid);
		ifs >> tmplon >> tmplat;
		//std::cout << tmplon << tmplat << std::endl;
		Geotag tmpGT = std::make_pair(tmplon, tmplat);
		Locations.insert(tmpGT);
		//std::cout << tmpGT << std::endl;
		std::set<std::string> s;
		std::getline(ifs, tmpString);
		std::istringstream iss(tmpString);
		while (iss >> tmpString2) {
			s.insert(tmpString2);
		}
		UserPosts[uid].push_back(Post(tmpGT, uid, s));
		LocToPost.insert(std::make_pair(std::make_pair(tmplon, tmplat),Post(tmpGT, uid, s)));
	}

	//Inputs;
	std::set<std::string> psi;
	psi.insert("Epulet1");
	psi.insert("Folyo");
	int m = 7;
	int sig = 3;


	//Algorithm STA (Algorithm 1)
	std::set<std::set<Geotag>> Rsig;

	C.resize(m + 2);
	for (auto L : Locations)
	{
		auto temp = std::set<Geotag>();
		temp.insert(L);
		C[0].insert(temp);
	}

	std::vector<std::set<std::set<Geotag>>> F;
	F.resize(m);

	Upsi = identifyRelevantUsers(psi);

	for (int i = 0; i < m; ++i) {
		for (auto L : C[i]) {
			//std::cout << L;
			computeSupports(L, psi);
			if (rw_sup[L] >= sig) {
				if (i == 1) {
					markObject(Pic, L);
					
				}
				F[i].insert(L);
				if (sup[L] >= sig)
					Rsig.insert(L);
			}
		}
		//std::cout << "i: "<<i <<" F: " << F[i];
		candidateGeneration(F[i], i + 2);
	}

	std::cout << "Rsig size: " << Rsig.size() << std::endl;
	std::cout << Rsig << std::endl;

	//std::cout << *(Rsig.begin()->begin());
	cv::imshow("Grad", Pic);
	cv::waitKey(0);
	return 0;
}
