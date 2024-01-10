#ifndef POINTCLOUDCALLBACK_H
#define POINTCLOUDCALLBACK_H

#include "trtCenterPoint.h"

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
        float max_x_range;
        float max_y_range;
        float min_x_range;
        float min_y_range;
        float max_z_range;
        float min_z_range;
        float ransac_distance_threshold;
        float ransac_max_iterations;
};

#endif