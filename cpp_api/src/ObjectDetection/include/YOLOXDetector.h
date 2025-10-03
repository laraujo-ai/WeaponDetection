#ifndef YOLOXDETECTOR_H
#define YOLOXDETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

#include "BaseModel.hpp"

struct Detection {
    float x1, y1, x2, y2;
    float score;
    int class_id;
};

class YOLOXDetector : public IBaseModel<cv::Mat, std::vector<Detection>> {
public:
    YOLOXDetector(const std::string& model_path, int num_threads = 1);
    ~YOLOXDetector() = default;

    std::vector<Detection> detect(const cv::Mat& image, float score_thr = 0.25f, float nms_thr = 0.45f);

protected:
    std::vector<Ort::Value> preprocess(const cv::Mat& input) override;
    std::vector<Detection> postprocess(std::vector<Ort::Value>& output_tensors) override;

private:
    std::pair<std::vector<float>, float> preprocessImage(const cv::Mat& ori_frame);
    std::vector<Detection> postprocessOutputs(const float* outputs, size_t output_size, float ratio,
                                               float score_threshold, float nms_threshold);
    void xywh_to_xyxy(std::vector<float>& boxes, float ratio);
    std::vector<int> nms(const std::vector<float>& boxes, const std::vector<float>& scores, float nms_thr);
    std::vector<Detection> multiclass_nms_class_agnostic(const std::vector<float>& boxes,
                                                          const std::vector<std::vector<float>>& scores,
                                                          float nms_thr, float score_thr);
    std::vector<int64_t> input_shape_;
    int target_h_;
    int target_w_;

    float score_threshold_;
    float nms_threshold_;
    float ratio_;
    std::vector<Ort::Float16_t> input_data_fp16_;
};

#endif // YOLOXDETECTOR_H