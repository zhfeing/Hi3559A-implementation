#ifndef __INC_BASIC_FUNCTION_H__
#define __INC_BASIC_FUNCTION_H__

#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <memory>
#include <hi_nnie.h>
#include <mpi_nnie.h>
#include <hi_comm_svp.h>
#include "resource_manager.hpp"

// contain information of a binary large object
class BlobInfo
{
public:
    BlobInfo(SVP_BLOB_TYPE_E en_type, HI_U32 stride, HI_U32 width, HI_U32 height, HI_U32 channel, HI_U32 batch_size = 1)
        :_en_type(en_type),             /* SVP_BLOB_TYPE_E */
         _stride(stride),               /* in byte: 32, 64, or 256 times odd integer */
         _width(width),                 /* shape */
         _height(height),               /* shape */
         _channel(channel),             /* shape */
         _batch_size(batch_size)        /* shape */
    { }
    
    SVP_BLOB_TYPE_E en_type()const { return _en_type; }
    HI_U32 stride()const { return _stride; }
    HI_U32 width()const { return _width; }
    HI_U32 height()const { return _height; }
    HI_U32 channel()const { return _channel; }
    HI_U32 batch_size()const { return _batch_size; }
private:
    HI_U32 _stride;
    HI_U32 _width;
    HI_U32 _height;
    HI_U32 _channel;
    HI_U32 _batch_size;
    SVP_BLOB_TYPE_E _en_type;
};

#ifdef __ARM_ARCH

// init arm system, useless while simulating
void system_init();

#endif // __ARM_ARCH


/* 
* warning: buffer is managed by "m_ptr rather" than "mem_info", 
* so user must hold m_ptr until mem_info is not required
*/
void buffer_to_mem_info(const ResourceManager<char> &m_ptr, SVP_MEM_INFO_S &mem_info);

// get wk file buffer from .wk file
void get_wk_file(const std::string &file_name, ResourceManager<char> &contents);

// load module and write result to pstModel
void load_module(SVP_SRC_MEM_INFO_S &stModelBuf, SVP_NNIE_MODEL_S &stModel);

// return task buffer size
HI_U32 get_task_buffer_size(SVP_NNIE_MODEL_S &stModel, HI_U32 max_input_num = 1, HI_U32 max_bounding_box_num = 0);

// calculate forward path
void calculate_forward(
    SVP_NNIE_HANDLE &svp_nnie_handle,
    SVP_SRC_BLOB_S &ast_scr_ptr,
    SVP_SRC_BLOB_S &ast_dst_ptr,
    SVP_NNIE_MODEL_S &stModel,
    SVP_NNIE_FORWARD_CTRL_S &forward_ctrl
);

// alloc BLOB object and return as blob_s
/*
*  WARNING: blob_s do not hold resource, it is blob_ptr that holds it, 
*  so keep blob_ptr till you want to release the resource
*/
void alloc_binary_large_obj(BlobInfo &blob_info, ResourceManager<HI_U8> &resource, 
    SVP_SRC_BLOB_S &blob_s);

// load image to resource_ptr
void load_image(const std::string &img_path, BlobInfo &blob_info, ResourceManager<HI_U8> &src_resource);

// unload model
void unload_model(SVP_NNIE_MODEL_S &stModel);

#endif /* __INC_BASIC_FUNCTION_H__ */
