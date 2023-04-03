// DON'T HAVE TO TOUCH
#include "readCSV.h"

void read_csv(std::string file_names, std::vector<double> &time, std::vector<double> &acc, std::vector<double> &pos, std::vector<double>&vel_gt)
{
    std::ifstream filestream(file_names);
    if (filestream.is_open())
    {
        std::string line;
        std::getline(filestream, line);
        while (std::getline(filestream, line))
        {
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream linestream(line);
            std::string time_s, acc_s, pos_s, vel_gt_s;
            linestream >> time_s >> acc_s >> pos_s >> vel_gt_s;

            time.push_back(std::stod(time_s));
            acc.push_back(std::stod(acc_s));
            pos.push_back(std::stod(pos_s));
            vel_gt.push_back(std::stod(vel_gt_s));
        }
    }

    return;
}


void save_csv(std::string file_names, vector<double> hat_Y, vector<double> Y){
    std::ofstream filestream(file_names);

    if (filestream.is_open())
    {
        filestream << "hat_Y" << ", " << "Y" << endl;
         for (int i = 0 ; i < hat_Y.size(); ++i){
            filestream << hat_Y[i] << ", " << Y[i] << endl;
         }
    }

}