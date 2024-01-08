#include "pointCloudCallback.h"


pointCloudCallbackClass::pointCloudCallbackClass(trtParams& params) : 
        centerpoint(params),
        input_cloud(new pcl::PointCloud<pcl::PointXYZ>),
        range_filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>),
        ground_cloud(new pcl::PointCloud<pcl::PointXYZ>),
        non_ground_cloud(new pcl::PointCloud<pcl::PointXYZ>),
        inliers(new pcl::PointIndices),
        coefficients(new pcl::ModelCoefficients)
{
    // engin init
    params.setParams();
    centerpoint.init(params);

    ROS_INFO("Building and running a GPU inference engine for CenterPoint");

    if(!centerpoint.engineInitlization()) 
    {
        ROS_ERROR("centerpoint build failed");
    }
    else 
    {
        ROS_INFO("Centerpoint build successed");
    }

    ROS_INFO("Waiting for point cloud . . .");
}


void pointCloudCallbackClass::publishBoxes(const std::vector<Box>& predResult) 
{
    visualization_msgs::MarkerArray markerArray;

    for (size_t i = 0; i < predResult.size(); ++i) 
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "os_sensor"; // 修改为你的坐标系
        marker.header.stamp = ros::Time::now();
        marker.ns = "boxes";
        marker.id = i;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;

        marker.pose.position.x = predResult[i].x;
        marker.pose.position.y = predResult[i].y;
        marker.pose.position.z = predResult[i].z;

        marker.scale.x = predResult[i].l;
        marker.scale.y = predResult[i].h;
        marker.scale.z = predResult[i].w;

        marker.color.r = 1.0;
        marker.color.g = predResult[i].cls;
        marker.color.b = predResult[i].score;
        marker.color.a = 0.5;
        // marker.text = predResult[i].cls;

        marker.pose.orientation.z = predResult[i].theta;    //

        marker.lifetime = ros::Duration(0.2);

        if(!predResult[i].isDrop && predResult[i].score > 0.1 && predResult[i].w < 2.0 && predResult[i].l < 2.0 && predResult[i].h < 2.0)
        {
            markerArray.markers.push_back(marker);
        }
    }
    marker_pub.publish(markerArray);
    ROS_INFO("Detected %ld Objects.", markerArray.markers.size());
}


void pointCloudCallbackClass::pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr &input)
{
    // ====================点云滤波====================
    auto start = std::chrono::system_clock::now();
    pcl::fromROSMsg (*input, *input_cloud);

    // range filter
    range_filter.setInputCloud(input_cloud);
    range_filter.setFilterFieldName("x");
    range_filter.setFilterLimits(-10.0, 10.0);
    range_filter.filter(*range_filtered_cloud);

    range_filter.setInputCloud(range_filtered_cloud);
    range_filter.setFilterFieldName("y");
    range_filter.setFilterLimits(-10.0, 10.0);
    range_filter.filter(*range_filtered_cloud);

    range_filter.setInputCloud(range_filtered_cloud);
    range_filter.setFilterFieldName("z");
    range_filter.setFilterLimits(-0.7, 1.0);
    range_filter.filter(*range_filtered_cloud);

    // removal ground point
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(500);
    seg.setDistanceThreshold(0.1);

    seg.setInputCloud(range_filtered_cloud);
    seg.segment(*inliers, *coefficients);

    // extract ground cloud
    extract.setInputCloud(range_filtered_cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*ground_cloud);

    // extract non ground cloud
    extract.setNegative(true);
    extract.filter(*non_ground_cloud);

    // assign global cloud
    centerpoint.global_cloud = non_ground_cloud;

    // ====================publish non ground====================
    pcl::toROSMsg(*non_ground_cloud, non_ground_cloud_msg);
    non_grd_pub.publish(non_ground_cloud_msg);

    auto after_filter = std::chrono::system_clock::now();
    std::chrono::duration<double> filter_time = after_filter - start;
    ROS_DEBUG("filter time: %f", filter_time.count());
    
    // ====================centerpoint infer====================
    if (!centerpoint.infer()) ROS_ERROR("infer failed! ");

    auto after_infer = std::chrono::system_clock::now();
    std::chrono::duration<double> infer_time = after_infer - after_filter;
    std::chrono::duration<double> all_time = after_infer - start;
    ROS_DEBUG("infer time: %f", infer_time.count());
    ROS_DEBUG("all time of a frame: %f", all_time.count());

    // ====================publish boxes====================
    if(centerpoint.predResult.size() == 0) ROS_WARN("no boxes detected! ");
    publishBoxes(centerpoint.predResult);
}


int main(int argc, char** argv)
{
    ros::init (argc, argv, "my_pcl_subscriber");

    // Set the logger level
    if(ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info)) {ros::console::notifyLoggerLevelsChanged();}

    ros::NodeHandle nh;

    trtParams params;
    trtCenterPoint centerpoint(params);

    // 将centerpoint传入pointCloudCallbackClass类中, 以便在pointCloudCallback中使用
    pointCloudCallbackClass pointCloudCallback(params);
    
    // Subscriber and Publisher
    cloud_sub   = nh.subscribe<sensor_msgs::PointCloud2> ("/os_cloud_node/points", 1, &pointCloudCallbackClass::pointCloudCallback, &pointCloudCallback);
    non_grd_pub = nh.advertise<sensor_msgs::PointCloud2> ("/non_ground_points", 1);
    marker_pub  = nh.advertise<visualization_msgs::MarkerArray> ("/centerpoint/dets", 1);

    // Start a spinner with 4 threads
    ros::AsyncSpinner spinner(4);
    spinner.start();
    ros::waitForShutdown();
}