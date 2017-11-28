#include <iostream>
#include <set>
#include <map>
#include <vector>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>     /* srand, rand */
#include <ctime>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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
std::set<std::string> psi;

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
				if (dist(p.loc, l) < 0.05) {
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
std::map<std::string, cv::Vec3b> colors;
cv::Vec3b getColor(const std::string& s) {
	//return cv::Vec3b(255, 255, 255);
	if (s == "Folyo")
		return cv::Vec3b(225, 0, 20);
	if (s == "Epulet1")
		return cv::Vec3b(225, 0, 255);
	if (s == "Epulet2")
		return cv::Vec3b(225, 255, 0);
	return cv::Vec3b(255, 255, 255);
}
const int D = 1000;
void markObject(cv::Mat& Pic, const Geotag& gt)
{
	cv::circle(Pic, cv::Point(D*(gt.first), D*(gt.second)), 4, cv::Scalar(0,0,0),2);
	cv::circle(Pic, cv::Point(D*(gt.first), D*(gt.second)), 3, cv::Scalar(getColor(*(LocToPost.at(gt).keywords.begin()))),-1);
}

void generateData() {
	ofstream ofs("data.txt");
	int N = 1500;
	int usernum = 200;
	srand(time(nullptr));
	ofs << N << std::endl;
	for (int i = 0; i < N; i++) {
		int tosz = rand() % 4;
		if (tosz == 0) {
			//ofs << rand()%12 << ' ' << 0.8+0.01*((rand()%200) - 100) << ' ' << 0.5 + 0.01*((rand() % 200) - 100) <<" Epulet1\n";
			double alpha2, d;
			do {
				int alpha = rand() % 360;
				alpha2 = alpha * 2 * 3.14159 / 360;
				d = pow(0.015*((rand() % 200) - 100), 3);
			} while (!(d > 0.005 && 0.8 + sin(alpha2)*d < 1 && 0.8 + sin(alpha2)*d > 0 && 0.5 + cos(alpha2)*d < 1 && 0.5 + cos(alpha2)*d > 0));
			
			ofs << rand() % 12 << ' ' << 0.8 + sin(alpha2)*d << ' ' << 0.5 + cos(alpha2)*d << " Epulet1\n";
		}//Building 1
		else if (tosz == 1) {
			double alpha2, d;
			do {
				int alpha = rand() % 360;
				alpha2 = alpha * 2 * 3.14159 / 360;
				d = pow(0.015*((rand() % 200) - 100), 3);
			} while (!(0.4 + sin(alpha2)*d < 1 && 0.4 + sin(alpha2)*d > 0 && 0.8 + cos(alpha2)*d < 1 && 0.8 + cos(alpha2)*d > 0));
			ofs << rand() % 12 << ' ' << 0.4 + sin(alpha2)*d  << ' ' << 0.8 + cos(alpha2)*d << " Epulet2\n";
		}
		else{
			double alpha2, d, p1, p2;
			do {
				int alpha = rand() % 360;
				alpha2 = alpha * 2 * 3.14159 / 360;
				d = pow(0.03*((rand() % 200) - 100), 3);
				p1 = (((rand() % 10000) + 10000) % 10000) / 10000.0;
				p2 = 2*p1 - 2.0 / 5.0;
			} while (!(d > 0.009 && d < 0.8 &&p2 + sin(alpha2)*d < 1 && p2 + sin(alpha2)*d > 0 && p1 + cos(alpha2)*d < 1 && p1 + cos(alpha2)*d > 0));
			
			
			ofs << rand() % 12 << ' ' << p2 + sin(alpha2)*d << ' ' << p1 + cos(alpha2)*d << " Folyo\n";
		}
	}
	ofs.close();
}

void initObjects(cv::Mat& Pic) {
	
	//FOLYO
	for (int i = 0; i < D; i++)
		for (int j = 0; j < D; j++)
		{
			if (abs((-i + 2 * j - D*2.0 / 5.0)) < D / 5.0 * 0.7)
				Pic.at<cv::Vec3b>(j, i) = cv::Vec3b(225, 0, 20);
			else
				Pic.at<cv::Vec3b>(j, i) = cv::Vec3b(10, 125, 70);
		}
	cv::circle(Pic, cv::Point(D*0.8, D*0.5), 20, cv::Scalar(255, 0, 255), -1);
	cv::circle(Pic, cv::Point(D*0.4, D*0.8), 20, cv::Scalar(255, 255, 0), -1);
}

int main()
{
	//Inputs;

	psi.insert("Epulet1");
	psi.insert("Folyo");
	int m = 1;
	int sig = 10;


	cv::Mat Pic = cv::Mat_<cv::Vec3b>(D, D);
	//cv::imshow("Grad", Pic);
	//cv::waitKey(0);
	generateData();
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
		if(!LocToPost.count(std::make_pair(tmplon, tmplat)) || psi.count(*s.begin()))
			LocToPost.insert(std::make_pair(std::make_pair(tmplon, tmplat),Post(tmpGT, uid, s)));
	}
	initObjects(Pic);

	//Algorithm STA (Algorithm 1)
	std::set<std::set<Geotag>> Rsig;

	C.resize(m + 2);
	for (auto L : Locations)
	{
		auto temp = std::set<Geotag>();
		temp.insert(L);
		C[0].insert(temp);
		markObject(Pic, L);
	}
	cv::imshow("Grad", Pic);
	cv::waitKey(0);
	initObjects(Pic);

	std::vector<std::set<std::set<Geotag>>> F;
	F.resize(m);

	Upsi = identifyRelevantUsers(psi);
	int maxsup = 0;
	std::set<Geotag> maxLoc;
	for (int i = 0; i < m; ++i) {
		for (auto L : C[i]) {
			//std::cout << L;
			computeSupports(L, psi);
			if (rw_sup[L] >= sig) {
				for(auto& l : L) {
					if(psi.count(*(LocToPost.at(l).keywords.begin())))
						markObject(Pic, l);
				}
				F[i].insert(L);
				if (sup[L] >= sig)
					Rsig.insert(L);
				if (sup[L] >= maxsup && L.size() == 1) { maxsup = sup[L]; maxLoc = L; }
			}
		}
		//std::cout << "i: "<<i <<" F: " << F[i];
		if(i != m-1)
			candidateGeneration(F[i], i + 2);
	}

	std::cout << "Rsig size: " << Rsig.size() << std::endl;
	//std::cout << Rsig << std::endl;
	std::cout << maxLoc << std::endl;
	cv::circle(Pic, cv::Point(D*maxLoc.begin()->first, D*maxLoc.begin()->second), 8, cv::Scalar(0, 200, 240), -1);
	cv::circle(Pic, cv::Point(D*maxLoc.begin()->first, D*maxLoc.begin()->second), 9, cv::Scalar(0, 0, 0), 2);

	//std::cout << *(Rsig.begin()->begin());
	cv::imshow("Grad", Pic);
	cv::waitKey(0);
	return 0;
}
