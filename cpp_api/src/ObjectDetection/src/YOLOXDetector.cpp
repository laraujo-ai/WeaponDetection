#include "YOLOXDetector.h"
#include <iostream>



YOLOXDetector::YOLOXDetector(const std::string& model_path, int num_threads)
    : IBaseModel<cv::Mat, std::vector<Detection>>(model_path, num_threads),
      score_threshold_(0.25f),
      nms_threshold_(0.45f),
      ratio_(1.0f)
{
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
    auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    input_shape_ = tensor_info.GetShape();

    target_h_ = static_cast<int>(input_shape_[2]);
    target_w_ = static_cast<int>(input_shape_[3]);
}

std::vector<Detection> YOLOXDetector::detect(const cv::Mat& image, float score_thr, float nms_thr) {
    score_threshold_ = score_thr;
    nms_threshold_ = nms_thr;
    return run(image);
}

std::vector<Ort::Value> YOLOXDetector::preprocess(const cv::Mat& input) {
    auto [input_data, ratio] = preprocessImage(input);
    ratio_ = ratio;

    // Convert float32 to float16 as required by the model
    std::vector<Ort::Float16_t> input_data_fp16(input_data.size());
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data_fp16[i] = Ort::Float16_t(input_data[i]);
    }
    input_data_fp16_ = std::move(input_data_fp16);

    std::vector<int64_t> input_shape = {1, 3, target_h_, target_w_};

    auto tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
        memory_info_,
        input_data_fp16_.data(),
        input_data_fp16_.size(),
        input_shape.data(),
        input_shape.size()
    );

    std::vector<Ort::Value> tensors;
    tensors.push_back(std::move(tensor));
    return tensors;
}

std::pair<std::vector<float>, float> YOLOXDetector::preprocessImage(const cv::Mat& ori_frame) {
    cv::Mat padded_img(target_h_, target_w_, CV_8UC3, cv::Scalar(114, 114, 114));

    float r = std::min(static_cast<float>(target_h_) / ori_frame.rows,
                       static_cast<float>(target_w_) / ori_frame.cols);

    int resized_h = static_cast<int>(ori_frame.rows * r);
    int resized_w = static_cast<int>(ori_frame.cols * r);
    cv::Mat resized_img;
    cv::resize(ori_frame, resized_img, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_LINEAR);

    resized_img.copyTo(padded_img(cv::Rect(0, 0, resized_w, resized_h)));

    cv::Mat float_img;
    padded_img.convertTo(float_img, CV_32F);

    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);
    std::vector<float> input_data(3 * target_h_ * target_w_);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(input_data.data() + c * target_h_ * target_w_,
                    channels[c].data,
                    target_h_ * target_w_ * sizeof(float));
    }

    return {input_data, r};
}

std::vector<Detection> YOLOXDetector::postprocess(std::vector<Ort::Value>& output_tensors) {
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    size_t output_size = 1;
    for (auto dim : output_shape) {
        output_size *= dim;
    }

    return postprocessOutputs(output_data, output_size, ratio_, score_threshold_, nms_threshold_);
}

std::vector<Detection> YOLOXDetector::postprocessOutputs(const float* outputs, size_t output_size,
                                                          float ratio, float score_threshold,
                                                          float nms_threshold) {
    std::vector<int> strides = {8, 16, 32};
    std::vector<std::pair<int, int>> grid_sizes;
    std::vector<std::vector<std::pair<float, float>>> grids;
    std::vector<std::vector<float>> expanded_strides;

    for (int stride : strides) {
        int hsize = target_h_ / stride;
        int wsize = target_w_ / stride;

        std::vector<std::pair<float, float>> grid;
        std::vector<float> strides_vec;

        for (int y = 0; y < hsize; ++y) {
            for (int x = 0; x < wsize; ++x) {
                grid.push_back({static_cast<float>(x), static_cast<float>(y)});
                strides_vec.push_back(static_cast<float>(stride));
            }
        }

        grids.push_back(grid);
        expanded_strides.push_back(strides_vec);
    }

    std::vector<std::pair<float, float>> all_grids;
    std::vector<float> all_strides;

    for (size_t i = 0; i < grids.size(); ++i) {
        all_grids.insert(all_grids.end(), grids[i].begin(), grids[i].end());
        all_strides.insert(all_strides.end(), expanded_strides[i].begin(), expanded_strides[i].end());
    }

    size_t num_predictions = all_grids.size();
    size_t num_attrs = 85;  // 4 (bbox) + 1 (objectness) + 80 (classes) -> this is coco default

    std::vector<float> boxes;
    std::vector<std::vector<float>> scores;

    for (size_t i = 0; i < num_predictions; ++i) {
        size_t idx = i * num_attrs;

        float cx = (outputs[idx] + all_grids[i].first) * all_strides[i];
        float cy = (outputs[idx + 1] + all_grids[i].second) * all_strides[i];

        float w = std::exp(outputs[idx + 2]) * all_strides[i];
        float h = std::exp(outputs[idx + 3]) * all_strides[i];

        boxes.push_back(cx);
        boxes.push_back(cy);
        boxes.push_back(w);
        boxes.push_back(h);

        float objectness = outputs[idx + 4];
        std::vector<float> class_scores;
        for (size_t c = 0; c < 80; ++c) {
            class_scores.push_back(objectness * outputs[idx + 5 + c]);
        }
        scores.push_back(class_scores);
    }

    xywh_to_xyxy(boxes, ratio);
    return multiclass_nms_class_agnostic(boxes, scores, nms_threshold, score_threshold);
}

void YOLOXDetector::xywh_to_xyxy(std::vector<float>& boxes, float ratio) {
    for (size_t i = 0; i < boxes.size(); i += 4) {
        float cx = boxes[i];
        float cy = boxes[i + 1];
        float w = boxes[i + 2];
        float h = boxes[i + 3];

        boxes[i] = (cx - w / 2.0f) / ratio;
        boxes[i + 1] = (cy - h / 2.0f) / ratio;
        boxes[i + 2] = (cx + w / 2.0f) / ratio;
        boxes[i + 3] = (cy + h / 2.0f) / ratio;
    }
}

std::vector<int> YOLOXDetector::nms(const std::vector<float>& boxes, const std::vector<float>& scores,
                                     float nms_thr) {
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&scores](int i1, int i2) {
        return scores[i1] > scores[i2];
    });

    std::vector<float> areas(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        float x1 = boxes[i * 4];
        float y1 = boxes[i * 4 + 1];
        float x2 = boxes[i * 4 + 2];
        float y2 = boxes[i * 4 + 3];
        areas[i] = (x2 - x1 + 1) * (y2 - y1 + 1);
    }

    std::vector<int> keep;
    while (!indices.empty()) {
        int idx = indices[0];
        keep.push_back(idx);

        if (indices.size() == 1) break;

        std::vector<int> new_indices;
        for (size_t i = 1; i < indices.size(); ++i) {
            int idx2 = indices[i];

            float x1 = std::max(boxes[idx * 4], boxes[idx2 * 4]);
            float y1 = std::max(boxes[idx * 4 + 1], boxes[idx2 * 4 + 1]);
            float x2 = std::min(boxes[idx * 4 + 2], boxes[idx2 * 4 + 2]);
            float y2 = std::min(boxes[idx * 4 + 3], boxes[idx2 * 4 + 3]);

            float w = std::max(0.0f, x2 - x1 + 1);
            float h = std::max(0.0f, y2 - y1 + 1);
            float inter = w * h;

            float iou = inter / (areas[idx] + areas[idx2] - inter);

            if (iou <= nms_thr) {
                new_indices.push_back(idx2);
            }
        }

        indices = new_indices;
    }

    return keep;
}

std::vector<Detection> YOLOXDetector::multiclass_nms_class_agnostic(const std::vector<float>& boxes,
                                                                     const std::vector<std::vector<float>>& scores,
                                                                     float nms_thr, float score_thr) {
    size_t num_boxes = scores.size();
    std::vector<int> cls_inds;
    std::vector<float> cls_scores;

    for (const auto& score_vec : scores) {
        auto max_it = std::max_element(score_vec.begin(), score_vec.end());
        int cls_idx = std::distance(score_vec.begin(), max_it);
        float max_score = *max_it;

        cls_inds.push_back(cls_idx);
        cls_scores.push_back(max_score);
    }

    std::vector<float> valid_boxes;
    std::vector<float> valid_scores;
    std::vector<int> valid_cls_inds;

    for (size_t i = 0; i < num_boxes; ++i) {
        if (cls_scores[i] > score_thr) {
            valid_boxes.push_back(boxes[i * 4]);
            valid_boxes.push_back(boxes[i * 4 + 1]);
            valid_boxes.push_back(boxes[i * 4 + 2]);
            valid_boxes.push_back(boxes[i * 4 + 3]);
            valid_scores.push_back(cls_scores[i]);
            valid_cls_inds.push_back(cls_inds[i]);
        }
    }

    if (valid_scores.empty()) {
        return std::vector<Detection>();
    }

    std::vector<int> keep = nms(valid_boxes, valid_scores, nms_thr);

    std::vector<Detection> detections;
    for (int idx : keep) {
        Detection det;
        det.x1 = valid_boxes[idx * 4];
        det.y1 = valid_boxes[idx * 4 + 1];
        det.x2 = valid_boxes[idx * 4 + 2];
        det.y2 = valid_boxes[idx * 4 + 3];
        det.score = valid_scores[idx];
        det.class_id = valid_cls_inds[idx];
        detections.push_back(det);
    }

    return detections;
}
