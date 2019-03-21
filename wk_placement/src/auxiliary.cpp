#include "auxiliary.h"
#include <iostream>

void report_error(const std::string &description, HI_S32 error_code)
{
    std::cout << description << " failed: error code: 0x" << std::hex << error_code << std::endl;
    exit(-1);
}