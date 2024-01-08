#ifndef TRT_CENTERPOINT_H
#define TRT_CENTERPOINT_H

#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <chrono>

#include "centerpoint.h"
#include "config.h"


using namespace std;

const string PACKAGE_PATH = ros::package::getPath("pcl_subscriber");


struct trtParams: public Params{
    void setParams() {
        // initialize sample parameters 
        if(PACKAGE_PATH.empty()) {
            ROS_ERROR("PACKAGE_PATH is empty ! use direct path !");
            // this->pfeSerializedEnginePath = "/home/CenterPoint/catkin_points_workspack/src/pcl_subscriber/models/pfe_fp.engine";
            // this->rpnSerializedEnginePath = "/home/CenterPoint/catkin_points_workspack/src/pcl_subscriber/models/rpn_fp.engine";
            this->pfeSerializedEnginePath = "/home/CenterPoint/catkin_points_workspack/src/pcl_subscriber/models/pfe_int8.engine";
            this->rpnSerializedEnginePath = "/home/CenterPoint/catkin_points_workspack/src/pcl_subscriber/models/rpn_int8.engine";
            ROS_INFO("pfe Engine Path: %s", this->pfeSerializedEnginePath.c_str());
            ROS_INFO("rpn Engine Path: %s", this->rpnSerializedEnginePath.c_str());
        }
        else {
            ROS_INFO("PACKAGE_PATH: %s", PACKAGE_PATH.c_str());
            this->pfeSerializedEnginePath = PACKAGE_PATH + "/models/pfe_fp.engine";
            this->rpnSerializedEnginePath = PACKAGE_PATH + "/models/rpn_fp.engine";
        }

        this->load_engine = true;

        // Input Output Names, according to TASK_NUM
        this->pfeInputTensorNames.push_back("input.1");
        this->rpnInputTensorNames.push_back("input.1");
        this->pfeOutputTensorNames.push_back("47");

        this->rpnOutputTensorNames["regName"]  = {"246"};
        this->rpnOutputTensorNames["rotName"] = {"258"};
        this->rpnOutputTensorNames["heightName"]={"250"};
        this->rpnOutputTensorNames["dimName"] = {"264"};
        this->rpnOutputTensorNames["scoreName"] = {"265"};
        this->rpnOutputTensorNames["clsName"] = {"266"};

        // Attrs
        
        this->batch_size = 1;

        ROS_INFO("pfeOnnxFilePath: %s", this->pfeOnnxFilePath.c_str());
        ROS_INFO("rpnOnnxFilePath: %s", this->rpnOnnxFilePath.c_str());
        ROS_INFO("pfeSerializedEnginePath: %s", this->pfeSerializedEnginePath.c_str());
        ROS_INFO("rpnSerializedEnginePath: %s", this->rpnSerializedEnginePath.c_str());
        ROS_INFO("savePath: %s", this->savePath.c_str());
        ROS_INFO("fp16: %d", this->fp16);
        ROS_INFO("load_engine: %d", this->load_engine);
        ROS_INFO("dlaCore: %d", this->dlaCore);
        ROS_INFO("batch_size: %d", this->batch_size);
    }
};


class trtCenterPoint : public CenterPoint {
    public:
        std::vector<Box> predResult;    // 存储预测结果
        pcl::PointCloud<pcl::PointXYZ>::Ptr global_cloud; // 存储点云数据

        template <typename T>
        using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

        std::shared_ptr<nvinfer1::ICudaEngine> getEngine() const { return mEngine; }
        std::shared_ptr<nvinfer1::ICudaEngine> getEngineRPN() const { return mEngineRPN; }
        void setEngine(std::shared_ptr<nvinfer1::ICudaEngine> engine) { mEngine = engine; }
        void setEngineRPN(std::shared_ptr<nvinfer1::ICudaEngine> engine) { mEngineRPN = engine; }


        trtCenterPoint() = default;
        trtCenterPoint(const trtParams params);
        void init(const trtParams params);
        std::shared_ptr<nvinfer1::ICudaEngine> buildFromSerializedEngine(std::string serializedEngineFile);
        bool engineInitlization ();
        bool infer();

    private:
        // device pointers 
        float* dev_scattered_feature_;
        float* dev_points_ ;
        int* dev_indices_;
        int* dev_score_idx_;
        long* dev_keep_data_;
        SampleUniquePtr<ScatterCuda> scatter_cuda_ptr_;

        // device pointers for preprocess
        int* p_bev_idx_; 
        int* p_point_num_assigned_;
        bool* p_mask_;
        int* bev_voxel_idx_; // H * W
        float* v_point_sum_;
        int* v_range_;
        int* v_point_num_;
        
        // host  variables for post process
        long* host_keep_data_;
        float* host_boxes_;
        int* host_label_;
        int* host_score_idx_;
        unsigned long long* mask_cpu;
        unsigned long long* remv_cpu;

        Params mParams;
        int BATCH_SIZE_ = 1;
        nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
        nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
        int mNumber{0};             //!< The number to classify
        std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
        std::shared_ptr<nvinfer1::ICudaEngine> mEngineRPN;

        // 重构infer函数
        // SampleUniquePtr<nvinfer1::IExecutionContext> context;
        // SampleUniquePtr<nvinfer1::IExecutionContext> contextRPN;
        float* points;
        cudaStream_t stream;
        


        float* processInput(pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud) {
            size_t point_count = pcl_cloud->points.size();

            // 伸长率，设置为零
            float elongation_rate = 0.0;
            // 创建一个动态数组存储点云数据
            float* point_cloud_data = new float[5 * point_count];

            pcl::PointCloud<pcl::PointXYZI> pclCloud;
            pcl::copyPointCloud(*pcl_cloud, pclCloud);

            for(size_t i = 0; i < point_count; ++i){
                point_cloud_data[5 * i] = pclCloud.points[i].x;
                point_cloud_data[5 * i + 1] = pclCloud.points[i].y;
                point_cloud_data[5 * i + 2] = pclCloud.points[i].z;
                // point_cloud_data[5 * i + 3] = pclCloud.points[i].intensity;
                point_cloud_data[5 * i + 3] = 0.0;
                point_cloud_data[5 * i + 4] = elongation_rate;
            }
            return point_cloud_data;
        }

};


#endif // TRT_CENTERPOINT_H