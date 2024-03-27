#include "trtCenterPoint.h"


trtCenterPoint::trtCenterPoint(const trtParams params)
    : CenterPoint(params),  
    mParams(params), 
    mEngine(nullptr), 
    mEngineRPN(nullptr), 
    BATCH_SIZE_(params.batch_size),
    global_cloud(new pcl::PointCloud<pcl::PointXYZ>)
{
    scatter_cuda_ptr_.reset(new ScatterCuda(PFE_OUTPUT_DIM, PFE_OUTPUT_DIM, BEV_W, BEV_H ));

    // mallocate a global memory for pointer
    GPU_CHECK(cudaMalloc((void**)&dev_points_, MAX_POINTS * POINT_DIM * sizeof(float)));
    GPU_CHECK(cudaMemset(dev_points_,0, MAX_POINTS * POINT_DIM * sizeof(float)));

    GPU_CHECK(cudaMalloc((void**)&dev_indices_,MAX_PILLARS * sizeof(int)));
    GPU_CHECK(cudaMemset(dev_indices_,0,MAX_PILLARS * sizeof(int)));

    /**
     * @brief : Create and Init Variables for PreProcess
     * 
     */
    GPU_CHECK(cudaMalloc((void**)& p_bev_idx_, MAX_POINTS * sizeof(int)));
    GPU_CHECK(cudaMalloc((void**)& p_point_num_assigned_, MAX_POINTS * sizeof(int)));
    GPU_CHECK(cudaMalloc((void**)& p_mask_, MAX_POINTS * sizeof(bool)));
    GPU_CHECK(cudaMalloc((void**)& bev_voxel_idx_, BEV_H * BEV_W * sizeof(int)));

    GPU_CHECK(cudaMemset(p_bev_idx_, 0, MAX_POINTS * sizeof(int)));
    GPU_CHECK(cudaMemset(p_point_num_assigned_, 0, MAX_POINTS * sizeof(int)));
    GPU_CHECK(cudaMemset(p_mask_, 0, MAX_POINTS * sizeof(bool)));
    GPU_CHECK(cudaMemset(bev_voxel_idx_, 0, BEV_H * BEV_W * sizeof(int)));

    GPU_CHECK(cudaMalloc((void**)&v_point_sum_, MAX_PILLARS * 3 *sizeof(float)));
    GPU_CHECK(cudaMalloc((void**)&v_range_, MAX_PILLARS * sizeof(int)));
    GPU_CHECK(cudaMalloc((void**)&v_point_num_, MAX_PILLARS * sizeof(int)));


    GPU_CHECK(cudaMemset(v_range_,0, MAX_PILLARS * sizeof(int)));
    GPU_CHECK(cudaMemset(v_point_sum_, 0, MAX_PILLARS * 3 * sizeof(float)));

    /**
     * @brief : Create and Init Variables for PostProcess
     * 
     */
    GPU_CHECK(cudaMalloc((void**)&dev_score_idx_, OUTPUT_W * OUTPUT_H * sizeof(int)));
    GPU_CHECK(cudaMemset(dev_score_idx_, -1 , OUTPUT_W * OUTPUT_H * sizeof(int)));

    GPU_CHECK(cudaMallocHost((void**)& mask_cpu, INPUT_NMS_MAX_SIZE * DIVUP (INPUT_NMS_MAX_SIZE ,THREADS_PER_BLOCK_NMS) * sizeof(unsigned long long)));
    GPU_CHECK(cudaMemset(mask_cpu, 0 ,  INPUT_NMS_MAX_SIZE * DIVUP (INPUT_NMS_MAX_SIZE ,THREADS_PER_BLOCK_NMS) * sizeof(unsigned long long)));

    GPU_CHECK(cudaMallocHost((void**)& remv_cpu, THREADS_PER_BLOCK_NMS * sizeof(unsigned long long)));
    GPU_CHECK(cudaMemset(remv_cpu, 0 ,  THREADS_PER_BLOCK_NMS  * sizeof(unsigned long long)));

    GPU_CHECK(cudaMallocHost((void**)&host_score_idx_, OUTPUT_W * OUTPUT_H  * sizeof(int)));
    GPU_CHECK(cudaMemset(host_score_idx_, -1, OUTPUT_W * OUTPUT_H  * sizeof(int)));

    GPU_CHECK(cudaMallocHost((void**)&host_keep_data_, INPUT_NMS_MAX_SIZE * sizeof(long)));
    GPU_CHECK(cudaMemset(host_keep_data_, -1, INPUT_NMS_MAX_SIZE * sizeof(long)));

    GPU_CHECK(cudaMallocHost((void**)&host_boxes_, OUTPUT_NMS_MAX_SIZE * 9 * sizeof(float)));
    GPU_CHECK(cudaMemset(host_boxes_, 0 ,  OUTPUT_NMS_MAX_SIZE * 9 * sizeof(float)));

    GPU_CHECK(cudaMallocHost((void**)&host_label_, OUTPUT_NMS_MAX_SIZE * sizeof(int)));
    GPU_CHECK(cudaMemset(host_label_, -1, OUTPUT_NMS_MAX_SIZE * sizeof(int)));
}


void trtCenterPoint::init(const trtParams params)
{
    mParams = params;
    mEngine = nullptr;
    mEngineRPN = nullptr;
    BATCH_SIZE_ = params.batch_size;
}


std::shared_ptr<nvinfer1::ICudaEngine> trtCenterPoint::buildFromSerializedEngine(std::string serializedEngineFile)
{
    cout << "serializedEngineFile : " << serializedEngineFile << endl;
    std::vector<char> trtModelStream_;
    size_t size{0};
    std::ifstream file(serializedEngineFile, std::ios::binary);

    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0,file.beg);
        trtModelStream_.resize(size);
        file.read(trtModelStream_.data(), size);
        file.close() ;
    } else {
        ROS_ERROR("Failed to read serialized engine !");
        return nullptr;
    }
    
    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    
    if(!runtime) { 
        ROS_ERROR("Failed to create runtime !"); 
        return nullptr;
    }

    ROS_DEBUG("Create ICudaEngine  !");
    std::shared_ptr<nvinfer1::ICudaEngine>  engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(trtModelStream_.data(), size), 
        samplesCommon::InferDeleter());

    if (!engine)
    {
        ROS_ERROR("Failed to create engine !");
        return nullptr;
    }

    return engine;
}


bool trtCenterPoint::engineInitlization()
{
    ROS_DEBUG("Building pfe engine . . .");
    mEngine = buildFromSerializedEngine(mParams.pfeSerializedEnginePath);
    ROS_DEBUG("Building rpn engine . . .");
    mEngineRPN = buildFromSerializedEngine(mParams.rpnSerializedEnginePath);
    ROS_DEBUG("All has Built !");
    return true;
}


bool trtCenterPoint::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);
    samplesCommon::BufferManager buffersRPN(mEngineRPN);
    // Create RAII context on device

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    auto contextRPN = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngineRPN->createExecutionContext());
    
    if (!context || !contextRPN)
    {
        ROS_ERROR("Failed to create execution context !");
        return false;
    }

    // mParams.inputTensorNames :  [ voxels,  num_voxels, coords ]
    float* devicePillars = static_cast<float*>(buffers.getDeviceBuffer(mParams.pfeInputTensorNames[0]));

    int pointNum = global_cloud->size();
    points = processInput(global_cloud);
    ROS_DEBUG("Point cloud preprocess Done !");

    // Create cuda stream to profile this infer pipline 
    GPU_CHECK(cudaStreamCreate(&stream));

    // Doing preprocess 
    GPU_CHECK(cudaMemcpy(dev_points_, points, pointNum * POINT_DIM * sizeof(float), cudaMemcpyHostToDevice));

    preprocessGPU(dev_points_, devicePillars, dev_indices_, 
    p_mask_, p_bev_idx_,  p_point_num_assigned_,  bev_voxel_idx_, v_point_sum_,  v_range_,  v_point_num_,
        pointNum, POINT_DIM);

    // Memcpy from host input buffers to device input buffers
    // buffers.copyInputToDevice();
    // buffersRPN.copyInputToDevice();

    // Execute the inference work
    auto start_pfe = std::chrono::system_clock::now();
    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        ROS_WARN("Failed to execute inf !");
        return false;
    }
    auto after_pfe = std::chrono::system_clock::now();
    std::chrono::duration<double> pfe_infer_time = std::chrono::duration_cast<std::chrono::duration<double>>(after_pfe - start_pfe);
    ROS_DEBUG("pfe infer time: %f", pfe_infer_time.count());

    // Memcpy from device output buffers to host output buffers
    // buffers.copyOutputToHost();
    // buffersRPN.copyOutputToHost();
    
    dev_scattered_feature_ = static_cast<float*>(buffersRPN.getDeviceBuffer(mParams.rpnInputTensorNames[0]));

    GPU_CHECK(cudaMemset(dev_scattered_feature_, 0 ,  PFE_OUTPUT_DIM * BEV_W * BEV_H * sizeof(float)));

    scatter_cuda_ptr_->doScatterCuda(MAX_PILLARS, dev_indices_,static_cast<float*>(buffers.getDeviceBuffer(mParams.pfeOutputTensorNames[0])), 
                                                            //   static_cast<float*>(buffersRPN.getDeviceBuffer(mParamsRPN.inputTensorNames[0]) )) ;
                                                            dev_scattered_feature_);

    // ROS_INFO("Running RPN inference on device . . .");
    auto start_rpn = std::chrono::system_clock::now();
    status = contextRPN->executeV2( buffersRPN.getDeviceBindings().data());
    auto after_rpn = std::chrono::system_clock::now();
    std::chrono::duration<double> rpn_infer_time = std::chrono::duration_cast<std::chrono::duration<double>>(after_rpn - start_rpn);
    ROS_DEBUG("rpn infer time: %f", rpn_infer_time.count());
    if (!status)
    {
        ROS_WARN("Failed to execute RPN !");
        return false;
    }
    ROS_DEBUG("RPN inference Done !");
    // post process
    predResult.clear();
    ROS_DEBUG("Running postprocess on device . . .");
    postprocessGPU(buffersRPN, predResult, mParams.rpnOutputTensorNames,
                                            dev_score_idx_,
                                            mask_cpu,
                                            remv_cpu,
                                            host_score_idx_,
                                            host_keep_data_,
                                            host_boxes_,
                                            host_label_);

    delete[] points;
    cudaStreamDestroy(stream);
    ROS_DEBUG("Postprocess Done !");

    return true;
}