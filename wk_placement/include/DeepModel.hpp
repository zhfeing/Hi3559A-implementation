#ifndef __INC_DEEP_MODEL_HPP__
#define __INC_DEEP_MODEL_HPP__

#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <memory>
#include <hi_nnie.h>
#include <hi_comm_svp.h>
#include "basic_function.h"
#include "resource_manager.hpp"
#include "auxiliary.h"

using BlobType = SVP_BLOB_TYPE_E;

class DeepModel
{
public:
    DeepModel(const BlobInfo &scr_blob_info, const BlobInfo &dst_blob_info);
    ~DeepModel();
    void init();
    void open_wk_file(const std::string &wk_file_name);
    void load_input(const std::string &img_file_path);
    template <typename T>
    const T *predict();
    
private:
    void load_model();
    void alloc_auxiliary_buffer();
    void set_control_parameters();
    void malloc_scr_ptr();
    void malloc_dst_ptr();
private:
    //std::shared_ptr<char> wk_buf;
    ResourceManager<char> wk_resource;

    SVP_NNIE_MODEL_S nnie_model;

    SVP_MEM_INFO_S tmp_buf_info;
    SVP_MEM_INFO_S tsk_buf_info;

    ResourceManager<char> tmp_resource;
    ResourceManager<char> tsk_resource;

    SVP_NNIE_ID_E nnie_id;
    SVP_NNIE_FORWARD_CTRL_S forward_ctrl;
    SVP_NNIE_HANDLE svp_nnie_handle;

    BlobInfo scr_blob_info;
    ResourceManager<HI_U8> scr_resource;
    SVP_SRC_BLOB_S scr_blob_s;

    BlobInfo dst_blob_info;
    ResourceManager<HI_U8> dst_resource;
    SVP_SRC_BLOB_S dst_blob_s;

    bool is_initialized;
};

template <typename T>
const T *DeepModel::predict()
{
    if (is_initialized)
    {
        calculate_forward(svp_nnie_handle, scr_blob_s, dst_blob_s, nnie_model, forward_ctrl);
    }
    else
    {
        std::cout << "predict before initial" << std::endl;
        dst_resource.free();
    }
    T *p = reinterpret_cast<T*>(dst_resource.get_raw_ptr());
    return p;
}

#endif /* __DEEP_MODEL_HPP__ */
