// DON'T HAVE TO TOUCH
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <string>

using namespace std;

// csv 파일 입출력을 위해 작성한 helper 함수
void read_csv(string file_names, vector<vector<double>> &group_a, vector<vector<double>> &group_b);
void save_csv(std::string file_names, std::vector<std::vector<double>> group_a, std::vector<std::vector<double>> group_b, std::vector<double> label_a, std::vector<double> label_b, unordered_map<string, double> initWeight, unordered_map<string, double> finalWeight, int flag);