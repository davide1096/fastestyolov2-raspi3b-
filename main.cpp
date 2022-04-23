#include <lccv.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <yolofastestv2.h>
#include <cmath>

yoloFastestv2 yf2;
//float focal_length = 3.04;
char room[50] = "living_room"; // LOCATION: living room, corridor
char light[50] = "natural"; // LIGHT: natural, artificial low, artificial high
char text[256];

int main()
{

    yf2.init(false);
    yf2.loadModel("yolo-fastestv2-opt.param","yolo-fastestv2-opt.bin");

    std::cout<<"Press ESC to stop."<<std::endl;
    //timings
    std::chrono::steady_clock::time_point Tbegin, Tend, Tstart;
    float f, elapsed_time;
    float FPS[16];
    float approx_dist;
    float error_sum = 0;
    int i,Fcnt=0;
    int counter = 0;
    int dis_counter = 0;
    int N = 0;
    int false_counter = 0;
    char path[256];
    std::vector<float> known_dis = {0.0,1.5,2.0,2.5,3.0,3.5};
    std::vector<int> img_count = {};
    std::vector<float> errors = {};
    std::vector<int> falses = {};
    cv::Mat image;
    lccv::PiCamera cam;
    cam.options->video_width=1024;
    cam.options->video_height=768;
    cam.options->framerate=5;
    //cam.options->verbose=true;
    //cv::namedWindow("Video",cv::WINDOW_NORMAL);
    cam.startVideo();
    int ch=0;
    cv::VideoWriter video("output2.avi",cv::VideoWriter::fourcc('M','J','P','G'), 4, cv::Size(320,320));
    Tstart = std::chrono::steady_clock::now();
    while (ch!=27){
        if (ch==32){
            ch = 0;
            img_count.push_back(N);
            N = 0;
            errors.push_back(error_sum);
            error_sum = 0;
            falses.push_back(false_counter);
            false_counter = 0;
            dis_counter += 1;
            if (dis_counter > 5){
                break;
            }
            Tstart = std::chrono::steady_clock::now();
        }
        elapsed_time = std::chrono::duration_cast <std::chrono::milliseconds> (Tend - Tstart).count();
        if ((elapsed_time < 5000) || (elapsed_time>15000)){
            ch=cv::waitKey(10);
            Tend = std::chrono::steady_clock::now();
            continue;
        }
        Tbegin = std::chrono::steady_clock::now();
        if(!cam.getVideoFrame(image,1000)){
            std::cout<<"Timeout error"<<std::endl;
        }
        else{
            std::vector<TargetBox> boxes;
            cv::resize(image,image,cv::Size(320,320));
            sprintf(path, "./dataset/%s/%s/%.1f/%s_%s_%d.jpg", room, light, known_dis[dis_counter], room, light, counter);
            cv::imwrite(path, image);

            yf2.detection(image, boxes);
            approx_dist = yf2.drawObjects(image, boxes);
            if (known_dis[dis_counter] == 0.0){
                if (approx_dist > 0.0){
                    false_counter += 1;
                }
            } else {
                if (approx_dist == 0.0){
                    false_counter += 1;
                }
            }
            error_sum += std::pow((approx_dist - known_dis[dis_counter]),2);
            Tend = std::chrono::steady_clock::now();
            f = std::chrono::duration_cast <std::chrono::milliseconds> (Tend - Tbegin).count();
            if(f>0.0) FPS[((Fcnt++)&0x0F)]=1000.0/f;
            for(f=0.0, i=0;i<16;i++){ f+=FPS[i]; }
            putText(image, cv::format("FPS %0.2f", f/16),cv::Point(10,20),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(0, 0, 255));

            video.write(image);
            cv::imshow("Video",image);

            ch=cv::waitKey(10);
            counter += 1;
            N += 1;

        }

    }

    std::ofstream file;
    sprintf(path, "./dataset/%s/%s/report.txt", room, light);
    file.open(path, std::ios::out | std::ios::trunc);
    sprintf(text,"ROOM: %s, LIGHT: %s, %d Images captured\n",room,light,counter);
    file << text;
    for (int i=0; i<6;i++){
        sprintf(text, "-%0.2f   error=%0.2f, num_images=%d, falses=%d\n", known_dis[i],
                std::sqrt(errors[i]/img_count[i]), img_count[i], falses[i]);
        file << text;
    }
    file.close();
    cam.stopVideo();
    video.release();
    cv::destroyWindow("Video");
    return 0;
}
