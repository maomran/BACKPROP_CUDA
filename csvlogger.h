#ifndef CSV_LOGGER_H
#define CSV_LOGGER_H

#include <string>
#include <cstdlib>
#include <cstdio>
#include <fstream>

class CSVLogger {

public:
    int epoch;
    std::ofstream csvFile;
    CSVLogger(std::string fileName);
    ~CSVLogger();

    void logEpoch(double trainingLoss, double trainingAccuracy, 
    			  double testLoss, double testAccuracy,
                  double totalForwardTime, double totalBackwardTime, 
                  double batchForwardTime, double batchBackwardTime);
};

#endif /* CSV_LOGGER_H */
