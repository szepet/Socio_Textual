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
