#include "readCSV.h"



void read_csv(std::string file_names, std::vector<std::vector<double>> &group_a, std::vector<std::vector<double>> &group_b){
    std::ifstream filestream(file_names);
    if(filestream.is_open()){
        int idx =0;
        std::string line;
        while(std::getline(filestream, line)){
            std::replace(line.begin(), line.end(), ',',' ');
            std::istringstream linestream(line);
            std::string x_data, y_data, label;
            linestream >> x_data >> y_data >> label;
            std::vector<double> res;
            res = {std::stod(x_data), std::stod(y_data), std::stod(label)};
            if (idx <100)
                group_a.push_back(res);
            else
                group_b.push_back(res);
            idx+=1;
        }
    }
    return;
}

void save_csv(std::string file_names, std::vector<std::vector<double>> group_a, std::vector<std::vector<double>> group_b, std::vector<double> label_a, std::vector<double> label_b,  unordered_map<string, double> initWeight, unordered_map<string, double> finalWeight){
    std::ofstream filestream(file_names);

    if(filestream.is_open()){
        filestream << "Initial weights" << ", " << initWeight["W[0]"] << ", "  << initWeight["W[1]"] << ", "  << initWeight["B"] << std::endl;
        filestream << "Final weights" << ", " << finalWeight["W[0]"] << ", "  << finalWeight["W[1]"] << ", "  << finalWeight["B"] << std::endl;
        filestream << "X0" <<", " << "X1" <<", " << "LABEL(TRUE)" <<", " << "LABEL(Inference)" << ", " << "Prob." << std::endl;
        for (int i =0 ; i < group_a.size(); ++i)
            filestream<< group_a[i][0] <<", "<< group_a[i][1] <<", "<< group_a[i][2]<<", "<< std::round(label_a[i]) << ", "<< label_a[i] << std::endl;

        for (int i =0 ; i < group_b.size(); ++i)
            filestream<< group_b[i][0] <<", "<< group_b[i][1] <<", "<< group_b[i][2] <<", "<< std::round(label_b[i])<< ", "<<label_b[i] << std::endl;

        
    }
}

