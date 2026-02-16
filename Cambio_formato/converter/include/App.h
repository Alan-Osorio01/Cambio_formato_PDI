#pragma once
#include <string>

class App {
public:
    int run(int argc, char** argv);

private:
    void printUsage(const std::string& exe) const;
};
