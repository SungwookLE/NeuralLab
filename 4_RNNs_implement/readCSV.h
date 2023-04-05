// DON'T HAVE TO TOUCH
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <string>
#include <iostream>

using namespace std;

// csv 파일 입출력을 위해 작성한 helper 함수
void read_csv(std::string file_names, std::vector<double> &time, std::vector<double> &acc, std::vector<double> &pos, std::vector<double> &vel_gt);
void save_csv(std::string file_names, vector<double> hat_Y, vector<double> Y);