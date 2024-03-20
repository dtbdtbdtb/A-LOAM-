// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#include <cmath>
#include <vector>
#include <string>
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

using std::atan2;
using std::cos;
using std::sin;

const double scanPeriod = 0.1;//扫描周期为固定0.1s

const int systemDelay = 0; //   系统初始化延迟
int systemInitCount = 0;
bool systemInited = false;
int N_SCANS = 0;//激光雷达线数，通过参数服务器传入，若未指定，默认为16
float cloudCurvature[400000];//点云曲率
int cloudSortInd[400000];//点云排序索引
int cloudNeighborPicked[400000];//自身或附近点被选为特征点标志
int cloudLabel[400000]; //点云标签
                        // Label 2: corner_sharp曲率大（角点）
                        // Label 1: corner_less_sharp, 包含Label 2（曲率稍微小的点，降采样角点）
                        // Label -1: surf_flat（平面点）
                        // Label 0: surf_less_flat， 包含Label -1，因为点太多，最后会降采样

bool comp (int i,int j) { return (cloudCurvature[i]<cloudCurvature[j]); }

ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubRemovePoints;
std::vector<ros::Publisher> pubEachScan;

bool PUB_EACH_LINE = false;

double MINIMUM_RANGE = 0.1; //允许的点云最近距离，过近的点滤除

template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                              pcl::PointCloud<PointT> &cloud_out, float thres)//移除点云中距离为thres的点
{
    if (&cloud_in != &cloud_out)
    {
        cloud_out.header = cloud_in.header;//保留头信息，包括seq（序列长度）、stamp（时间戳）、frame_id（坐标系）
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;//size_t表示无符号整数类型，等价于unsigned long long, 用于表示数组的大小

    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }
    if (j != cloud_in.points.size())
    {
        cloud_out.points.resize(j);
    }

    cloud_out.height = 1;//设置点云的行数，表示点云高度方向的分辨率。取值越大，分辨率越高，点云越精细。
    cloud_out.width = static_cast<uint32_t>(j);//width：类型为uint32_t，表示点云宽度（如果组织为图像结构），即一行点云的数量。
    cloud_out.is_dense = true;//is_dense：bool 类型，若点云中的数据都是有限的（不包含 inf/NaN 这样的值），则为 true，否则为 false。
}

void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
//************************************************************************************************************
    //数据预处理阶段    滤除NaN的点云及过近点云，求出每个点的所在线束及相对于初始点的时间间隔以补偿点云畸变
    //系统初始化部分，若设置了初始化时间systemDelay>0，则在指定时间内完成初始化
    if (!systemInited)
    { 
        systemInitCount++;
        if (systemInitCount >= systemDelay)
        {
            systemInited = true;
        }
        else
            return;
    }

    //作者自己设计的计时类，以构造函数为起始时间，以toc()函数为终止时间，并返回时间间隔(ms)
    TicToc t_whole;
    TicToc t_prepare;

    //每条扫描线上的可以计算曲率的点云点的起始索引和结束索引
    //分别用scanStartInd向量和scanEndInd向量数组记录，初始值都为0
    std::vector<int> scanStartInd(N_SCANS, 0);
    std::vector<int> scanEndInd(N_SCANS, 0);

    //  将ROS消息转换为PCL点云并滤除NaN的点云及移除距离过近的点云
    pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);//将ROS消息转换为PCL点云
    std::vector<int> indices;//记录过滤后的点云的索引

    pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);//该函数用于从点云数据中移除NaN（非数字）点。它接受一个输入点云和一个输出点云作为参数，并返回一个包含有效点的索引数组。
    removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE);

    //Velodyne是顺时针的，加负号变为逆时针，因为角度表示的正方向为逆时针，如何区分不同线束
    //确定水平角度范围，便于分配并对齐点云时间戳
    int cloudSize = laserCloudIn.points.size();
    float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
    float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                          laserCloudIn.points[cloudSize - 1].x) +
                   2 * M_PI;

    if (endOri - startOri > 3 * M_PI)
    {
        endOri -= 2 * M_PI;
    }
    else if (endOri - startOri < M_PI)
    {
        endOri += 2 * M_PI;
    }
    //printf("end Ori %f\n", endOri);
    //根据俯仰角确定点云所在线束，并将每一线束的点云单独存放，为后续提取特征点做准备
    bool halfPassed = false;
    int count = cloudSize;
    PointType point;//点云类型为xyzi
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
    for (int i = 0; i < cloudSize; i++)
    {
        point.x = laserCloudIn.points[i].x;
        point.y = laserCloudIn.points[i].y;
        point.z = laserCloudIn.points[i].z;

        float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
        int scanID = 0;

        if (N_SCANS == 16)
        {
            scanID = int((angle + 15) / 2 + 0.5);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 32)
        {
            scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 64)
        {   
            if (angle >= -8.83)
                scanID = int((2 - angle) * 3.0 + 0.5);
            else
                scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

            // use [0 50]  > 50 remove outlies 
            if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else
        {
            printf("wrong scan number\n");
            ROS_BREAK();
        }
        //printf("angle %f scanID %d \n", angle, scanID);

        float ori = -atan2(point.y, point.x);
        if (!halfPassed)
        { 
            if (ori < startOri - M_PI / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > startOri + M_PI * 3 / 2)
            {
                ori -= 2 * M_PI;
            }

            if (ori - startOri > M_PI)
            {
                halfPassed = true;
            }
        }
        else
        {
            ori += 2 * M_PI;
            if (ori < endOri - M_PI * 3 / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > endOri + M_PI / 2)
            {
                ori -= 2 * M_PI;
            }
        }

        float relTime = (ori - startOri) / (endOri - startOri);
        point.intensity = scanID + scanPeriod * relTime;//用强度表示线束编号，线束编号范围为[0, N_SCANS-1]，小数部分表示线束的相对时间。
        laserCloudScans[scanID].push_back(point); //这里的push_back函数用于向点云中添加一个点，使用的是点云类中的方法，而不是容器的push_back函数
    }
    
    cloudSize = count;
    printf("points size %d \n", cloudSize);

//*******************************************************************************************************
    //计算曲率，提取特征点
    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());//创建一个std::shared_ptr对象，指向一个新的点云实例，这样可以保证点云对象在其不再被任何其他对象引用时自动释放内存。
    //将每条扫描线上的点输入到laserCloud指向的点云,
    //并处理每个scan，将每条线的起始索引设置为第六个点，结束的索引为倒数第六个点，即：每条scan上面的点的起始 id 为前5个点不要  结束的id 为后5个点不要，因为这些点不满足求曲率所需要的前后五个点的条件
    //同时，将无序点云转换为有序点云laserCloud（线束顺序排列）
    for (int i = 0; i < N_SCANS; i++)
    { 
        scanStartInd[i] = laserCloud->size() + 5;//这里的laserCloud->size()函数用于获取点云中点的数量，返回值为点云的大小，即点云中点的数量。
        *laserCloud += laserCloudScans[i];
        scanEndInd[i] = laserCloud->size() - 6;
    }

    printf("prepare time %f \n", t_prepare.toc());

    //计算曲率，同一条扫描线上的取目标点左右两侧各5个点，分别与目标点的坐标作差平方和，得到的结果就是目标点的曲率
    for (int i = 5; i < cloudSize - 5; i++)//这里是以一帧的有序点云来计算曲率，因此在每个scan交界处的曲率不准确，可通过scanStartInd来选取
    { 
        float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
        float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
        float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;

        cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;//曲率
        cloudSortInd[i] = i;
        cloudNeighborPicked[i] = 0;// 索引为i的点有没有被选择为特征点
        cloudLabel[i] = 0;  //点的类型
                            // Label 2: corner_sharp曲率大（角点）
                            // Label 1: corner_less_sharp, 包含Label 2（曲率稍微小的点，降采样角点）
                            // Label -1: surf_flat（平面点）
                            // Label 0: surf_less_flat， 包含Label -1，因为点太多，最后会降采样
    }


    TicToc t_pts;
    //定义4个点云，分别用来存放特征点
    pcl::PointCloud<PointType> cornerPointsSharp;
    pcl::PointCloud<PointType> cornerPointsLessSharp;
    pcl::PointCloud<PointType> surfPointsFlat;
    pcl::PointCloud<PointType> surfPointsLessFlat;

    //提取特征点，论文中对每条扫描线分成了4个扇形，而代码中实际是分成了6份，在每份中寻找曲率最大的20个点为角点，对提取数据做了约束，最大点不大于2个，平面点不大于4个，剩下的均为次平面点；
    //为防止特征点过于集中，每提取一个特征点，就对该点和它附近的点的标志位设置未“已选中”，在循环提取时会跳过已提取的特征点，对于次极小面点采取下采样方式避免特征点扎堆。
    float t_q_sort = 0;//用于计算对曲率排序的时间
    for (int i = 0; i < N_SCANS; i++)
    {
        if( scanEndInd[i] - scanStartInd[i] < 6)
            continue;
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);//存放不太平整的点（后续需要降采样）
        //将每个scan分为6份
        for (int j = 0; j < 6; j++)
        {
            //每等份的起始和终止点索引
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6; 
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

            TicToc t_tmp;
            //根据点云中的点的曲率对点云中的点进行排序，顺序为从小到大
            //cloudSortInd存的是点云中点的索引，comp比较函数比较的是根据索引找到的对应点的曲率值，从而通过排序曲率，实现对索引的排序
            std::sort (cloudSortInd + sp, cloudSortInd + ep + 1, comp);
            // t_q_sort累计每个扇区曲率排序时间总和
            t_q_sort += t_tmp.toc();

            // 计算最大曲率点个数,选取极大角点（2个）和次极大角点（20个，包含极大角点）
            int largestPickedNum = 0;
            // 从最大曲率往最小曲率遍历，寻找边线点
            //如果自身或者相邻的点没有被选为特征点且曲率大于0.1，则认为该点可以进行特征判断
            for (int k = ep; k >= sp; k--)
            {
                int ind = cloudSortInd[k];//该索引即为该扇区曲率最大点所对应的索引 

                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] > 0.1)
                {

                    largestPickedNum++;
                    if (largestPickedNum <= 2)
                    {                        
                        cloudLabel[ind] = 2;//标记为曲率大角点
                        //将最大曲率点存入特征点云(cornerPointsSharp、cornerPointsLessSharp都存储)
                        cornerPointsSharp.push_back(laserCloud->points[ind]);
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    else if (largestPickedNum <= 20)
                    {                        
                        cloudLabel[ind] = 1;//标记次大角点
                        //将次大角点存入特征点云(cornerPointsLessSharp) 
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    else
                    {
                        break;//超过20个点，跳出循环
                    }
                    //对选中的特征角点进行操作
                    cloudNeighborPicked[ind] = 1;;//标记为已选中 
                    //对选中的特征角点的相邻点都进行标记以避免特征点过度集中，同时为防止新的边缘特征被滤除，对相邻点之间的距离进行判断，距离过大就不标记
                    //左侧五个点进行标记判断
                    for (int l = 1; l <= 5; l++)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    //右侧五个点进行标记判断
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }
            //从最小曲率往最大曲率遍历，寻找平面点
            int smallestPickedNum = 0;
            for (int k = sp; k <= ep; k++)
            {
                int ind = cloudSortInd[k];

                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] < 0.1)
                {

                    cloudLabel[ind] = -1; 
                    surfPointsFlat.push_back(laserCloud->points[ind]);

                    smallestPickedNum++;
                    //这里不区分平坦和比较平坦，因为剩下的点label默认是0,就是比较平坦
                    if (smallestPickedNum >= 4)
                    { 
                        break;
                    }

                    cloudNeighborPicked[ind] = 1;
                    for (int l = 1; l <= 5; l++)
                    { 
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }
            //选取次平面点，除了角点，剩下的（cloudLabel取值为0和-1）都是次平面点
            for (int k = sp; k <= ep; k++)
            {
                if (cloudLabel[k] <= 0)
                {
                    surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                }
            }
        }
        // 对每一条scan线上的次平面点进行一次降采样
        pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
        pcl::VoxelGrid<PointType> downSizeFilter;//使用PCL中的VoxelGrid滤波器对点云数据进行降采样处理。
        // 一般平坦的点比较多，所以这里做一个体素滤波
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);//设置输入点云(次平面点)
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);//设置体素大小
        downSizeFilter.filter(surfPointsLessFlatScanDS);//执行滤波操作并将结果保存到surfPointsLessFlatScanDS中。

        surfPointsLessFlat += surfPointsLessFlatScanDS;
    }
    printf("sort q time %f \n", t_q_sort);
    printf("seperate points time %f \n", t_pts.toc());

    //发布原始点云（将输入点云去噪转化为的有序点云）
    sensor_msgs::PointCloud2 laserCloudOutMsg;
    pcl::toROSMsg(*laserCloud, laserCloudOutMsg);//左为pcl点云数据，右为ROS点云数据
    laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;//输入点云消息的时间戳给输出点云消息
    laserCloudOutMsg.header.frame_id = "/camera_init";
    pubLaserCloud.publish(laserCloudOutMsg);

    //发布角点点云
    sensor_msgs::PointCloud2 cornerPointsSharpMsg;
    pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
    cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsSharpMsg.header.frame_id = "/camera_init";
    pubCornerPointsSharp.publish(cornerPointsSharpMsg);

    //发布次角点点云
    sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
    pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
    cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsLessSharpMsg.header.frame_id = "/camera_init";
    pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

    //发布平面点云
    sensor_msgs::PointCloud2 surfPointsFlat2;
    pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
    surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsFlat2.header.frame_id = "/camera_init";
    pubSurfPointsFlat.publish(surfPointsFlat2);

    //发布次平面点云
    sensor_msgs::PointCloud2 surfPointsLessFlat2;
    pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
    surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsLessFlat2.header.frame_id = "/camera_init";
    pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

    // pub each scan
    if(PUB_EACH_LINE)
    {
        for(int i = 0; i< N_SCANS; i++)
        {
            sensor_msgs::PointCloud2 scanMsg;
            pcl::toROSMsg(laserCloudScans[i], scanMsg);
            scanMsg.header.stamp = laserCloudMsg->header.stamp;
            scanMsg.header.frame_id = "/camera_init";
            pubEachScan[i].publish(scanMsg);
        }
    }

    printf("scan registration time %f ms *************\n", t_whole.toc());
    if(t_whole.toc() > 100)
        ROS_WARN("scan registration process over 100ms");
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "scanRegistration");// 初始化ROS节点，并指定节点名称为"scanRegistration"
    ros::NodeHandle nh;

    nh.param<int>("scan_line", N_SCANS, 16);// 通过nh参数服务器获取int类型的参数"scan_line"的值，如果参数不存在，则使用默认值N_SCANS，并将其设置为16

    nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1);

    printf("scan line number %d \n", N_SCANS);

    if(N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        printf("only support velodyne with 16, 32 or 64 scan line!");
        return 0;
    }
    
    // 创建一个订阅者subLaserCloud，用于订阅话题/velodyne_points的激光云点云数据，并将订阅到的数据传递给回调函数laserCloudHandler进行处理。
    // 数据类型为sensor_msgs::PointCloud2，最大订阅100个。
    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);

    // 创建一个发布者pubLaserCloud，用于发布处理后的激光云点云消息（类型sensor_msgs::PointCloud2）
    // 发布的话题名为/velodyne_cloud_2，发布频率为100Hz。
    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);

    pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100);

    pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100);

    pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100);

    pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100);

    pubRemovePoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 100);

    if(PUB_EACH_LINE)  // 发布每个扫描线的激光云点云消息,话题名为/laser_scanid_i,i为0~N_SCANS-1
    {
        for(int i = 0; i < N_SCANS; i++)
        {
            ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/laser_scanid_" + std::to_string(i), 100);
            pubEachScan.push_back(tmp);
        }
    }
    ros::spin();// 进入消息循环，处理接收到的消息。

    return 0;
}
