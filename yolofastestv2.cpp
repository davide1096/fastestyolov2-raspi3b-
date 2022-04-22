#include <math.h>
#include <algorithm>
#include <yolofastestv2.h>

const char* class_names[] = {
    "background", "person", "bicycle",
    "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "sofa", "pottedplant", "bed", "diningtable",
    "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
};

float intersection_area(const TargetBox& first, const TargetBox& second){

    if (first.x1 > second.x2 || first.x2 < second.x1 || first.y1 > second.y2 || first.y2 < second.y1){
        return 0.f; // no intersection
    }
    float inter_width = std::min(first.x2, second.x2) -  std::max(first.x1, second.x1);
    float inter_height = std::min(first.y2, second.y2) - std::max(first.y1, second.y1);

    return inter_width*inter_height;

}

bool scoreSort(TargetBox first, TargetBox second){

    return (first.score > second.score);

}

yoloFastestv2::yoloFastestv2(){
    numOutput = 2;
    numThreads = 4;
    numAnchor = 3;
    numCategory = 80;
    nmsThresh = 0.25;
    inputWidth = 352;
    inputHeight = 352;
    //anchor box w h
    std::vector<float> bias {12.64, 19.39, 37.88,51.48, 55.71, 138.31,
                             126.91, 78.23, 131.57, 214.55, 279.92, 258.87};

    anchor.assign(bias.begin(), bias.end());
}

yoloFastestv2::~yoloFastestv2(){

    ;

}

int yoloFastestv2::init(const bool use_vulkan_compute){

    net.opt.use_winograd_convolution = true;
    net.opt.use_sgemm_convolution = true;
    net.opt.use_int8_inference = true;
    net.opt.use_vulkan_compute = use_vulkan_compute;
    net.opt.use_fp16_packed = true;
    net.opt.use_fp16_storage = true;
    net.opt.use_fp16_arithmetic = true;
    net.opt.use_int8_storage = true;
    net.opt.use_int8_arithmetic = true;
    net.opt.use_packing_layout = true;
    net.opt.use_shader_pack8 = false;
    net.opt.use_image_storage = false;

    //net.opt.use_bf16_storage = true;

    return 0;

}

int yoloFastestv2::loadModel(const char* paramPath, const char* binPath){

    net.load_param(paramPath);
    net.load_model(binPath);

    printf("Ncnn model init success!\n");

    return 0;
}

int yoloFastestv2::interHandle(std::vector<TargetBox>& tmpBoxes, std::vector<TargetBox>& dstBoxes){

    std::vector<int> picked;
    sort(tmpBoxes.begin(), tmpBoxes.end(), scoreSort);

    for(size_t i = 0; i < tmpBoxes.size(); i++){
        int keep = 1;
        for (size_t j = 0; j < picked.size(); j++){

            float inter_area = intersection_area(tmpBoxes[i], tmpBoxes[picked[j]]);
            float union_area = tmpBoxes[i].area() + tmpBoxes[picked[j]].area() - inter_area;
            float iu_ratio = inter_area/union_area;

            if (iu_ratio > nmsThresh && tmpBoxes[i].cate == tmpBoxes[picked[j]].cate){
                keep = 0;
                break;
            }
        }

        if (keep){
            picked.push_back(i);
        }

    }

    for(size_t i=0; i < picked.size(); i++){
        dstBoxes.push_back(tmpBoxes[picked[i]]);
    }

    return 0;

}
int yoloFastestv2::getCategory(const float* values, int index, int& category, float& score){

    float tmp = 0;
    float objScore = values[4*numAnchor + index];

    for(int i=0; i < numCategory; i++){
        float clsScore = values[4*numAnchor + numAnchor + i];
        clsScore *= objScore;

        if (clsScore > tmp){
            score = clsScore;
            category = i;
            tmp = clsScore;
        }
    }
    return 0;
}

int yoloFastestv2::predHandle(const ncnn::Mat* out, std::vector<TargetBox>& dstBoxes,
                                      const float scaleW, const float scaleH, const float thresh){

        for(int i=0; i < numOutput; i++){
            int stride;
            int outW, outH, outC;

            outH = out[i].c;
            outW = out[i].h;
            outC = out[i].w;

            assert(inputHeight/outH == inputWidth/outW);
            stride = inputHeight/outH;

             for (int h = 0; h < outH; h++) {
                const float* values = out[i].channel(h);

                for(int w=0; w < outW; w++){

                    for(int b=0; b < numAnchor; b++){
                        TargetBox tmpBox;
                        int category = -1;
                        float score = -1;

                        getCategory(values, b, category, score);

                        if (score > thresh){
                            float bcx, bcy, bw, bh;
                            bcx = ((values[b * 4 + 0] * 2. - 0.5) + w) * stride;
                            bcy = ((values[b * 4 + 1] * 2. - 0.5) + h) * stride;
                            bw = pow((values[b * 4 + 2] * 2.), 2) * anchor[(i * numAnchor * 2) + b * 2 + 0];
                            bh = pow((values[b * 4 + 3] * 2.), 2) * anchor[(i * numAnchor * 2) + b * 2 + 1];

                            tmpBox.x1 = (bcx - 0.5 * bw) * scaleW;
                            tmpBox.y1 = (bcy - 0.5 * bh) * scaleH;
                            tmpBox.x2 = (bcx + 0.5 * bw) * scaleW;
                            tmpBox.y2 = (bcy + 0.5 * bh) * scaleH;
                            tmpBox.score = score;
                            tmpBox.cate = category;

                            dstBoxes.push_back(tmpBox);
                        }
                    }
                    values += outC;
                }
             }
        }
        return 0;
}

int yoloFastestv2::detection(const cv::Mat srcImg, std::vector<TargetBox>& dstBoxes, const float thresh){

    dstBoxes.clear();

    float scaleW = (float)srcImg.cols/(float)inputWidth;
    float scaleH = (float)srcImg.rows/(float)inputHeight;

     //resize of input image data
    ncnn::Mat inputImg = ncnn::Mat::from_pixels_resize(srcImg.data, ncnn::Mat::PIXEL_BGR,\
                                                       srcImg.cols, srcImg.rows, inputWidth, inputHeight);

    //Normalization of input image data
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    inputImg.substract_mean_normalize(mean_vals, norm_vals);

    //creat extractor
    ncnn::Extractor ex = net.create_extractor();
    ex.set_num_threads(numThreads);

    //set input tensor
    ex.input("input.1", inputImg);

    //forward
    ncnn::Mat out[2];
    ex.extract("794", out[0]); //22x22
    ex.extract("796", out[1]); //11x11

    std::vector<TargetBox> tmpBoxes;

    predHandle(out, tmpBoxes, scaleW, scaleH, thresh);

    //NMS
    interHandle(tmpBoxes, dstBoxes);

    return 0;

}

int yoloFastestv2::drawObjects(cv::Mat& cvImg, const std::vector<TargetBox>& boxes){
    for(size_t i = 0; i < boxes.size(); i++){
        char text[256];
        int pixel_height;
        float distance;
        if ((boxes[i].cate+1)==1){
            pixel_height = (boxes[i].y2-boxes[i].y1);
            distance = (1.73*2714.3*320/2464)/pixel_height;
            sprintf(text, "%s %.1f%% Approx d=%.2fm", class_names[boxes[i].cate+1], boxes[i].score * 100, distance);
        }
        else{
            sprintf(text, "%s %.1f%%", class_names[boxes[i].cate+1], boxes[i].score * 100);
        }


        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = boxes[i].x1;
        int y = boxes[i].y1 - label_size.height - baseLine;
        if (y < 0) y = 0;
        if (x + label_size.width > cvImg.cols) x = cvImg.cols - label_size.width;

        cv::rectangle(cvImg, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(cvImg, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

        cv::rectangle (cvImg, cv::Point(boxes[i].x1, boxes[i].y1),
                       cv::Point(boxes[i].x2, boxes[i].y2), cv::Scalar(255,0,0));

    }
    return 0;
}
