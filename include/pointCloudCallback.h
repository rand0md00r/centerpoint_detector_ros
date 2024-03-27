#ifndef POINTCLOUDCALLBACK_H
#define POINTCLOUDCALLBACK_H

#include "trtCenterPoint.h"
#include <tf/transform_listener.h>

// Global variables
ros::Subscriber cloud_sub;
ros::Publisher non_grd_pub;
ros::Publisher marker_pub;
ros::Publisher range_pub;



class pointCloudCallbackClass
{
    public:
        pointCloudCallbackClass(trtParams& params);
        void publishBoxes(const std::vector<Box>& predResult);
        void publishRange();
        void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& input);
        void paramServerInit(ros::NodeHandle& nh);
        void timerCallback(const ros::TimerEvent& event);

    private:
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud;
        pcl::PointCloud<pcl::PointXYZ>::Ptr range_filtered_cloud;
        pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud;
        pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground_cloud;

        pcl::PassThrough<pcl::PointXYZ> range_filter;
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        pcl::PointIndices::Ptr inliers;
        pcl::ModelCoefficients::Ptr coefficients;
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        
        sensor_msgs::PointCloud2 non_ground_cloud_msg;

        trtCenterPoint centerpoint;

        // ParamServer params;
        ros::NodeHandle private_nh;
        double max_x_range;
        double max_y_range;
        double min_x_range;
        double min_y_range;
        double max_z_range;
        double min_z_range;
        double ransac_distance_threshold;
        double ransac_max_iterations;

        // 用定时器设定频率
        ros::Timer timer;

        // 坐标变换
        tf::TransformListener tf_listener;
        bool has_camera_init;
};

#endif