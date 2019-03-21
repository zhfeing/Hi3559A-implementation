#ifndef __INC_AUXILIARY_H__
#define __INC_AUXILIARY_H__

#pragma once

#include <hi_type.h>
#include <string>

class bad_out_of_range {};
class bad_alloc {};
class UseWithoutInit {};

void report_error(const std::string &description, HI_S32 error_code);


#endif // !__INC_AUXILIARY_H__

