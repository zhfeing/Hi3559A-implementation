#include "basic_function.h"
#include "resource_manager.hpp"
#include <thread>
#include <random>
#include "auxiliary.h"


#ifdef __ARM_ARCH

#include <fstream>
#include <iostream>

#else

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#endif // __ARM_ARCH


// init system
#ifdef __ARM_ARCH
#include <hi_comm_vb.h>
#include <mpi_vb.h>
#include <hi_comm_sys.h>
#include <mpi_sys.h>
#include "auxiliary.h"

void system_init()
{
    HI_S32 s32Ret;

    HI_MPI_SYS_Exit();
    HI_MPI_VB_Exit();
    VB_CONFIG_S struVbConf;
    memset(&struVbConf, 0, sizeof(VB_CONFIG_S));

    struVbConf.u32MaxPoolCnt = 10;
    struVbConf.astCommPool[1].u64BlkSize = 768 * 576 * 2;
    struVbConf.astCommPool[1].u32BlkCnt = 1;

    s32Ret = HI_MPI_VB_SetConfig((const VB_CONFIG_S *)&struVbConf);
    if (HI_SUCCESS != s32Ret)
    {
        report_error("HI_MPI_VB_SetConfig", s32Ret);
    }
    s32Ret = HI_MPI_VB_Init();
    if (HI_SUCCESS != s32Ret)
    {
        report_error("HI_MPI_VB_Init", s32Ret);
    }
    s32Ret = HI_MPI_SYS_Init();
    if (HI_SUCCESS != s32Ret)
    {
        report_error("HI_MPI_SYS_Init", s32Ret);
    }
}

#endif



void buffer_to_mem_info(const ResourceManager<char> &resource, SVP_MEM_INFO_S &mem_info)
{
    if (resource.is_null())
    {
        std::cout << "try to malloc from a nullptr" << std::endl;
        exit(-1);
    }
    mem_info.u64PhyAddr = resource.get_physic_address();
    mem_info.u64VirAddr = resource.get_virtual_address();
    mem_info.u32Size = resource.size();
}

void get_wk_file(const std::string &file_name, ResourceManager<char> &contents)
{
	std::ifstream wk_file(file_name, std::ios::binary | std::ios::in);
	if (!wk_file.is_open())
	{
		std::cout << "open .wk file failed" << std::endl;
		exit(-1);
	}
	std::streambuf * pbuf = wk_file.rdbuf();
	size_t size = pbuf->pubseekoff(0, wk_file.end);
	pbuf->pubseekoff(0, wk_file.beg);       // rewind
    contents.malloc(size);
	pbuf->sgetn(contents.get_raw_ptr(), size);
	wk_file.close();
}

void load_module(SVP_SRC_MEM_INFO_S &stModelBuf, SVP_NNIE_MODEL_S &stModel)
{
	// load module
	HI_S32 res;
	if (HI_SUCCESS != (res = HI_MPI_SVP_NNIE_LoadModel(&stModelBuf, &stModel)))
	{
		report_error("load_module", res);
	}
}

HI_U32 get_task_buffer_size(SVP_NNIE_MODEL_S &stModel, HI_U32 max_input_num, HI_U32 max_bounding_box_num)
{
	HI_U32 au32TskBufSize;
	HI_S32 res = HI_MPI_SVP_NNIE_GetTskBufSize(
		max_input_num,			/*max input num*/
		max_bounding_box_num,	/*max Bbox num*/
		&stModel,				/*model from load_model*/
		&au32TskBufSize,		/*task relate auxiliary buffer array*/
		stModel.u32NetSegNum	/*seg num*/
	);
	if (HI_SUCCESS != res)
	{
		report_error("get_task_buffer_size", res);
	}
	return au32TskBufSize;
}

void alloc_binary_large_obj(BlobInfo &blob_info, ResourceManager<HI_U8> &resource, 
    SVP_SRC_BLOB_S &blob_s)
{
    resource.malloc(
        blob_info.stride()*blob_info.height()*blob_info.channel()*blob_info.batch_size(), 
        false
    );

#ifdef __ARM_ARCH
    // resource.arm_flush_cache();
#endif // __ARM_ARCH

    blob_s.enType = blob_info.en_type();
    blob_s.u32Num = blob_info.batch_size();
    blob_s.u32Stride = blob_info.stride();
    blob_s.u64PhyAddr = resource.get_physic_address();
    blob_s.u64VirAddr = resource.get_virtual_address();
    blob_s.unShape.stWhc.u32Chn = blob_info.channel();
    blob_s.unShape.stWhc.u32Height = blob_info.height();
    blob_s.unShape.stWhc.u32Width = blob_info.width();
}

void calculate_forward(
    SVP_NNIE_HANDLE &svp_nnie_handle, 
    SVP_SRC_BLOB_S &ast_scr_ptr, 
    SVP_SRC_BLOB_S &ast_dst_ptr, 
    SVP_NNIE_MODEL_S &stModel, 
    SVP_NNIE_FORWARD_CTRL_S &forward_ctrl
)
{
    HI_S32 res = HI_MPI_SVP_NNIE_Forward(
        &svp_nnie_handle,       /* nnie handle */
        &ast_scr_ptr,      /* mulity node input */
        &stModel,               /* pst model */
        &ast_dst_ptr,      /* mulity node output */
        &forward_ctrl,          /* contrl structure */
        HI_TRUE
    );
    if (HI_SUCCESS != res)
    {
        report_error("calculate_forward", res);
    }

    HI_BOOL is_finish = HI_FALSE;

    do
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        res = HI_MPI_SVP_NNIE_Query(forward_ctrl.enNnieId, svp_nnie_handle, &is_finish, HI_TRUE);
        
    } while (HI_ERR_SVP_NNIE_QUERY_TIMEOUT == res);
    if (HI_SUCCESS != res)
    {
        report_error("HI_MPI_SVP_NNIE_Query", res);
    }
}

void load_image(const std::string &img_path, BlobInfo &blob_info, ResourceManager<HI_U8> &src_resource)
{
#ifdef __ARM_ARCH
    std::ifstream img_file(img_path);
    if (!img_file.is_open())
    {
        std::cout << "failed to open image file" << std::endl;
        exit(-1);
    }

    int pixel;
    for (unsigned c = 0; c < blob_info.channel(); c++)
    {
        for (unsigned i = 0; i < blob_info.height(); i++)
        {
            for (unsigned j = 0; j < blob_info.width(); j++)
            {
                img_file >> pixel;
                src_resource[c*blob_info.stride()*blob_info.height() + i * blob_info.stride() + j] = static_cast<HI_U8>(pixel);
            }
        }
    }
    
    img_file.close();
#else
    cv::Mat img_file = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    if (img_file.empty())
    {
        std::cout << "failed to open image file" << std::endl;
        exit(-1);
    }
    for (unsigned i = 0; i < blob_info.height(); i++)
    {
        for (unsigned j = 0; j < blob_info.width(); j++)
        {
            src_resource[i * blob_info.stride() + j] = img_file.at<HI_U8>(i, j);
        }
    }
#endif
}

void unload_model(SVP_NNIE_MODEL_S &stModel)
{
    HI_S32 res = HI_MPI_SVP_NNIE_UnloadModel(&stModel);
    if(HI_SUCCESS != res)
    {
        report_error("unload_model", res);
    }
}



