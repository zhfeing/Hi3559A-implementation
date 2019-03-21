#include "DeepModel.hpp"
#include <cmath>
#include <ctime>

#ifdef __ARM_ARCH

#include <sys/time.h>
#include <unistd.h>

#else

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#endif // __ARM_ARCH

using namespace std;
// using namespace cv;


int main()
{
#ifdef __ARM_ARCH
    system_init();
#endif
    // set basic parameter
    BlobInfo scr_blob_info(
        BlobType::SVP_BLOB_TYPE_U8,
        512,    /* stride */
        448,    /* width */
        256,    /* height */
        3,      /* channel */
        1       /* batch_size */
    );
    BlobInfo dst_blob_info(
        BlobType::SVP_BLOB_TYPE_S32, 
        1792,    /* stride */
        448,     /* width */
        256,     /* height */
        2,       /* channel */
        1        /* batch_size */
    );
    
    // set file path
    string wk_file_name = "./caffe_converted_chip.wk";
    string pred_file_name = "test_imgs/pred.txt";
    string img_file_path = "./imgs/";


    DeepModel deep_model(scr_blob_info, dst_blob_info);
    // load .wk file
    deep_model.open_wk_file(wk_file_name);

    deep_model.init();
    cout << "init works done!" << endl;
    //ifstream true_pred(pred_file_name);
    //if (!true_pred.is_open())
    //{
    //    cout << "open true pred failed" << endl;
    //    exit(-1);
    //}

    for(int img_id = 1; img_id < 6; img_id++)
    {
        
        string img_file_name = img_file_path + std::to_string(img_id) + ".txt";
        deep_model.load_input(img_file_name);

        float true_result;

#ifdef __ARM_ARCH
        struct timeval start, end;
        gettimeofday(&start, NULL);
#endif // __ARM_ARCH

        const HI_S32 *output = deep_model.predict<HI_S32>();


#ifdef __ARM_ARCH
        gettimeofday(&end, NULL);
        unsigned long long time_cost = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
        cout << "cost: " << time_cost / 1000.0 << "ms" << endl;
#endif // __ARM_ARCH

        ofstream img_out("img_out_" + to_string(img_id) + ".txt");
        for (int i = 0; i < 2 * 448 * 256; i++)
        {
            img_out << output[i] / 4096.0 << " ";
        }

    }
    return 0;
}
