// DON'T HAVE TO TOUCH
#include "readCSV.h"

void read_csv(std::string file_names, std::vector<std::vector<double>> &group_a, std::vector<std::vector<double>> &group_b)
{
    std::ifstream filestream(file_names);
    if (filestream.is_open())
    {
        int idx = 0;
        std::string line;
        while (std::getline(filestream, line))
        {
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream linestream(line);
            std::string x_data, y_data, label;
            linestream >> x_data >> y_data >> label;
            std::vector<double> res;
            res = {std::stod(x_data), std::stod(y_data), std::stod(label)};
            if (idx < 100)
                group_a.push_back(res);
            else
                group_b.push_back(res);
            idx += 1;
        }
    }
    return;
}

void save_csv(std::string file_names, std::vector<std::vector<double>> group_a, std::vector<std::vector<double>> group_b, std::vector<double> label_a, std::vector<double> label_b, unordered_map<string, double> initWeight, unordered_map<string, double> finalWeight, int flag = 1)
{

    std::ofstream filestream(file_names);

    if (filestream.is_open())
    {

        if (flag == 1)
        {
            double W_0, W_1, B;

            filestream << "Initial weights"
                       << ", ";
            for (auto v : initWeight)
            {
                if (v.first == "W[0]")
                    W_0 = v.second;
                else if (v.first == "W[1]")
                    W_1 = v.second;
                else if (v.first == "B")
                    B = v.second;
            }
            filestream << W_0 << ", " << W_1 << ", " << B << std::endl;

            filestream << "Final weights"
                       << ", ";
            for (auto v : finalWeight)
            {
                if (v.first == "W[0]")
                    W_0 = v.second;
                else if (v.first == "W[1]")
                    W_1 = v.second;
                else if (v.first == "B")
                    B = v.second;
            }
            filestream << W_0 << ", " << W_1 << ", " << B << std::endl;
        }
        else if (flag == 2)
        {
            double W_o1_0, W_o1_1, B_o;
            double W_h1_0, W_h1_1, B_h1;
            double W_h2_0, W_h2_1, B_h2;

            filestream << "Initial weights"
                       << ", ";
            for (auto v : initWeight)
            {
                if (v.first == "W_o1[0]")
                    W_o1_0 = v.second;
                else if (v.first == "W_o1[1]")
                    W_o1_1 = v.second;
                else if (v.first == "B_o1")
                    B_o = v.second;

                else if (v.first == "W_h1[0]")
                    W_h1_0 = v.second;
                else if (v.first == "W_h1[1]")
                    W_h1_1 = v.second;
                else if (v.first == "B_h1")
                    B_h1 = v.second;

                else if (v.first == "W_h2[0]")
                    W_h2_0 = v.second;
                else if (v.first == "W_h2[1]")
                    W_h2_1 = v.second;
                else if (v.first == "B_h2")
                    B_h2 = v.second;
            }
            filestream << W_o1_0 << ", " << W_o1_1 << ", " << B_o
                       << ", " << W_h1_0 << ", " << W_h1_1 << ", " << B_h1
                       << ", " << W_h2_0 << ", " << W_h2_1 << ", " << B_h2 << std::endl;

            filestream << "Final weights"
                       << ", ";
            for (auto v : finalWeight)
            {
                if (v.first == "W_o1[0]")
                    W_o1_0 = v.second;
                else if (v.first == "W_o1[1]")
                    W_o1_1 = v.second;
                else if (v.first == "B_o1")
                    B_o = v.second;

                else if (v.first == "W_h1[0]")
                    W_h1_0 = v.second;
                else if (v.first == "W_h1[1]")
                    W_h1_1 = v.second;
                else if (v.first == "B_h1")
                    B_h1 = v.second;

                else if (v.first == "W_h2[0]")
                    W_h2_0 = v.second;
                else if (v.first == "W_h2[1]")
                    W_h2_1 = v.second;
                else if (v.first == "B_h2")
                    B_h2 = v.second;
            }
            filestream << W_o1_0 << ", " << W_o1_1 << ", " << B_o
                       << ", " << W_h1_0 << ", " << W_h1_1 << ", " << B_h1
                       << ", " << W_h2_0 << ", " << W_h2_1 << ", " << B_h2 << std::endl;
        }

        filestream << "X0"
                   << ", "
                   << "X1"
                   << ", "
                   << "LABEL(TRUE)"
                   << ", "
                   << "LABEL(Inference)"
                   << ", "
                   << "Prob." << std::endl;
        for (int i = 0; i < group_a.size(); ++i)
            filestream << group_a[i][0] << ", " << group_a[i][1] << ", " << group_a[i][2] << ", " << std::round(label_a[i]) << ", " << label_a[i] << std::endl;

        for (int i = 0; i < group_b.size(); ++i)
            filestream << group_b[i][0] << ", " << group_b[i][1] << ", " << group_b[i][2] << ", " << std::round(label_b[i]) << ", " << label_b[i] << std::endl;
    }
}
