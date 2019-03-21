#include "DeepModel.hpp"
#include "auxiliary.h"



DeepModel::DeepModel(const BlobInfo &scr_blob_info, const BlobInfo &dst_blob_info)
    :wk_resource("nnie_model"), tmp_resource("tmp_resource"), tsk_resource("task_resource"), 
    scr_resource("scr_resource"), dst_resource("dst_resource"), 
    is_initialized(false), nnie_id(SVP_NNIE_ID_0), 
    scr_blob_info(scr_blob_info), dst_blob_info(dst_blob_info)
{

}

DeepModel::~DeepModel()
{
    if(nnie_model.stBase.u32Size != 0)
        unload_model(nnie_model);
}

void DeepModel::open_wk_file(const std::string &wk_file_name)
{
    if (!wk_resource.is_null())
    {
        std::cout << "wk file has already been opened" << std::endl;
        is_initialized = false;
    }
    get_wk_file(wk_file_name, wk_resource);
}

void DeepModel::load_model()
{
    SVP_SRC_MEM_INFO_S pstModelBuf =
    {
        wk_resource.get_physic_address(),   // pyhsic address
        wk_resource.get_virtual_address(),  // virtual address
        wk_resource.size()
    };
    load_module(pstModelBuf, nnie_model);
}

void DeepModel::alloc_auxiliary_buffer()
{
    // get required buffer size
    HI_U32 tmp_buffer_size = nnie_model.u32TmpBufSize;
    HI_U32 tsk_buffer_size = get_task_buffer_size(nnie_model);

    // alloc required buffer, cache only takes effect when __ARM_ARCH is defined
    tmp_resource.malloc(tmp_buffer_size, false);
    tsk_resource.malloc(tsk_buffer_size, false);

#ifdef __ARM_ARCH
    // tmp_resource.arm_flush_cache();
    // tsk_resource.arm_flush_cache();
#endif // __ARM_ARCH

    buffer_to_mem_info(tmp_resource, tmp_buf_info);
    buffer_to_mem_info(tsk_resource, tsk_buf_info);

}

void DeepModel::set_control_parameters()
{
    // config forward contrl structure
    forward_ctrl.u32SrcNum = scr_blob_info.batch_size();       /* input node num, [1, 16] */
    forward_ctrl.u32DstNum = scr_blob_info.batch_size();       /* output node num, [1, 16]*/
    forward_ctrl.u32NetSegId = 0;           /* net segment index running on NNIE */
    forward_ctrl.enNnieId = nnie_id;        /* device target which running the seg*/ 
    forward_ctrl.stTmpBuf = tmp_buf_info;   /* auxiliary temp mem */
    forward_ctrl.stTskBuf = tsk_buf_info;   /* auxiliary task mem */
}

void DeepModel::init()
{
    if (wk_resource.is_null())
    {
        std::cout << "wk file not loaded" << std::endl;
        throw UseWithoutInit();
    }
    // unload loaded model
    if(nnie_model.stBase.u32Size != 0)
    {
        unload_model(nnie_model);
        nnie_model.stBase.u32Size = 0;
    }
    load_model();
    alloc_auxiliary_buffer();
    set_control_parameters();
    malloc_scr_ptr();
    malloc_dst_ptr();
    is_initialized = true;
}

void DeepModel::malloc_scr_ptr()
{
    // malloc scr ptr
    alloc_binary_large_obj(scr_blob_info, scr_resource, scr_blob_s);
}
void DeepModel::malloc_dst_ptr()
{
    // malloc dst ptr
    alloc_binary_large_obj(dst_blob_info, dst_resource, dst_blob_s);
}

void DeepModel::load_input(const std::string &img_file_path)
{
    if (is_initialized)
    {
        load_image(img_file_path, scr_blob_info, scr_resource);
    }
    else
    {
        std::cout << "can not load input without initial" << std::endl;
        throw UseWithoutInit();
    }
}
