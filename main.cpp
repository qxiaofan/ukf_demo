#include "utils.h"
#include "readData.h"
#include "opencv2/opencv.hpp"

//功能
//本demo主要展示了ukf对于非线性系统的滤波过程。
//欢迎关注我们的公众号：[3D视觉工坊](https://mp.weixin.qq.com/s/xyGndcupuK1Zzmv1AJA5CQ)

//「3D视觉工坊」技术交流群已经成立，目前大约有12000人，方向主要涉及3D视觉、CV&深度学习、SLAM、三维重建、点云后处理、自动驾驶、CV入门、三维测量、VR/AR、3D人脸识别、医疗影像、缺陷检测、行人重识别、目标跟踪、视觉产品落
//地、视觉竞赛、车牌识别、硬件选型、学术交流、求职交流、ORB-SLAM系列源码交流、深度估计等。工坊致力于干货输出，不做搬运工，为计算机视觉领域贡献自己的力量！欢迎大家一起交流成长~

//添加小助手微信：*CV_LAB*，备注学校/公司+姓名+研究方向即可加入工坊一起学习进步。

//3D视觉研习社QQ群：574432628

//data, UKF
#include <random>
#include<UKF.h>
class UKF2DPoint : public UKF {
public:
    Matrix state_function (Matrix s)
    {
        Matrix state(4,1);
        state(0,0) = s(0,0)+s(2,0);	// x position in 2D point
        state(1,0) = s(1,0)+s(3,0);	// y position in 2D point
        state(2,0) = s(2,0);	// velocity in x
        state(3,0) = s(3,0);	// velocity in y
        return state;
    }

    Matrix measurement_function (Matrix m)
    {
        Matrix measurement(2,1);
        measurement(0,0) = m(0,0);	// measured x position in 2D point
        measurement(1,0) = m(1,0);	// measured y position in 2D point
        return measurement;
    }
};

int main(int argc,char **argv)
{
    //std::ofstream cmdoutfile("../output.txt");
    //std::cout.rdbuf(cmdoutfile.rdbuf());

    std::string file_data = argv[1];
    std::vector<cv::Point2f> dataPnts;
    PF::readData(file_data,dataPnts);

    float xRange = 800.0f;
    cv::Mat img((int)xRange,(int)xRange,CV_8UC3);

    cv::namedWindow("ukf");

    unsigned int n = 4;
    unsigned int m = 2;

    UKF2DPoint tracked_point;
    tracked_point.n = n;
    tracked_point.m = m;

    Matrix I4(4,4); //4x4 Identity Matrix
    I4(0,0) = 1;
    I4(1,1) = 1;
    I4(2,2) = 1;
    I4(3,3) = 1;
    Matrix I2(m,m); //2x2 Identity Matrix
    I2(0,0) = 1;
    I2(1,1) = 1;

    Matrix s(n,1); //initial state
    s(0,0) = 1;
    s(1,0) = 1;
    s(2,0) = 0;
    s(3,0) = 0;

    Matrix  x = s; //initial state
    const double q=0.1;	//std of process. "smoothness". lower the value, smoother the curve
    const double r=0.1;	//std of measurement. "tracking". lower the value, faster the track
    tracked_point.P = I4;	// state covriance
    tracked_point.Q = (q*q) * I4;	// covariance of process	(size must be nxn)
    tracked_point.R = (r*r) * I2;	// covariance of measurement (size must be mxm)

    std::vector<cv::Point2f> vcurrent_pos,vukf_pos;

    for(unsigned int k = 0; k < dataPnts.size();k++)
    {
        cv::Point2f measPt = cv::Point2f(dataPnts[k].x*100,
                                         dataPnts[k].y*100 );
        s(0,0) = dataPnts[k].x*100;	// measured x position in 2D point m->cm
        s(1,0) = dataPnts[k].y*100;	// measured y position in 2D point m->cm

        vcurrent_pos.push_back(cv::Point2f(s(0,0),s(1,0)));

        Matrix z = tracked_point.measurement_function(s); //make measurements

        tracked_point.ukf(x,z);

        cv::Point2f statePt= cv::Point2f(x(0,0),x(1,0));

        vukf_pos.push_back(statePt);


        //Clear screen
        img = cv::Scalar::all(100);

//      for(size_t i = 0; i < vcurrent_pos.size() - 1; i++)
//      {
//          cv::line(img,vcurrent_pos[i],vcurrent_pos[i + 1],cv::Scalar(255,255,0),1);
//      }

        for(size_t i = 0; i < vukf_pos.size() - 1; i++)
        {
            cv::line(img,vukf_pos[i],vukf_pos[i + 1],cv::Scalar(0,255,0),1);
        }

        s = tracked_point.state_function(s); // update process with artificial increment
        drawCross(img,statePt,cv::Scalar(255,255,255),5);
        drawCross(img,measPt,cv::Scalar(0,0,255),5);
        cv::Mat imgshow = cv::Mat::zeros(640,640,CV_8UC3);
        cv::resize(img,imgshow,cv::Size(500,500));
        cv::imshow("ukf",img);
        cv::waitKey(30);
    }
    
    system("pause");
    return 0;
}

