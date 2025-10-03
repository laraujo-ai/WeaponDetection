#include "ObjectDetection/include/YOLOXDetector.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <random>

struct Args {
    std::string media_link;
    std::string model_path;
    std::vector<std::string> class_names;
    float nms_thr = 0.45f;
    float score_thr = 0.25f;
    bool show = false;
    std::string out_file = "output.mp4";
};

class Visualizer {
public:
    Visualizer(const std::vector<std::string>& class_names) : class_names_(class_names) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);

        for (const auto& name : class_names) {
            colors_[name] = cv::Scalar(dis(gen), dis(gen), dis(gen));
        }
    }

    cv::Mat drawDetections(const cv::Mat& image, const std::vector<Detection>& detections, float alpha = 0.25f) {
        cv::Mat result = image.clone();

        if (detections.empty()) {
            return result;
        }

        int img_h = image.rows;
        int img_w = image.cols;

        for (const auto& det : detections) {
            int x1 = static_cast<int>(std::max(0.0f, std::min(det.x1, static_cast<float>(img_w))));
            int y1 = static_cast<int>(std::max(0.0f, std::min(det.y1, static_cast<float>(img_h))));
            int x2 = static_cast<int>(std::max(0.0f, std::min(det.x2, static_cast<float>(img_w))));
            int y2 = static_cast<int>(std::max(0.0f, std::min(det.y2, static_cast<float>(img_h))));

            if (x2 <= x1 || y2 <= y1) continue;

            std::string class_name = getClassName(det.class_id);
            cv::Scalar color = getColor(class_name);

            drawTranslucentBox(result, x1, y1, x2, y2, color, alpha);
            cv::rectangle(result, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
            drawLabel(result, x1, y1, x2, y2, class_name, det.score, color, img_w, img_h);
        }

        return result;
    }

    void processImage(const std::string& image_path, YOLOXDetector& detector, const Args& args) {
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            throw std::runtime_error("Image not found: " + image_path);
        }

        auto detections = detector.detect(image, args.score_thr, args.nms_thr);
        cv::Mat vis_image = drawDetections(image, detections);

        size_t dot_pos = image_path.find_last_of('.');
        std::string out_path = image_path.substr(0, dot_pos) + "_detected.jpg";
        cv::imwrite(out_path, vis_image);

        if (args.show) {
            cv::imshow("Detections", vis_image);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }

        std::cout << "Saved output image to " << out_path << std::endl;
    }

    void processVideoLive(cv::VideoCapture& cap, YOLOXDetector& detector, const Args& args) {
        cv::namedWindow("Detections", cv::WINDOW_NORMAL);
        cv::Mat frame;

        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            auto detections = detector.detect(frame, args.score_thr, args.nms_thr);
            cv::Mat vis_frame = drawDetections(frame, detections);

            cv::imshow("Detections", vis_frame);
            if (cv::waitKey(1) == 'q') break;
        }

        cap.release();
        cv::destroyAllWindows();
    }

    void processVideoToFile(cv::VideoCapture& cap, YOLOXDetector& detector, const Args& args,
                           double fps, int width, int height) {
        cv::VideoWriter out(args.out_file, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps, cv::Size(width, height));
        cv::Mat frame;

        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            auto detections = detector.detect(frame, args.score_thr, args.nms_thr);
            cv::Mat vis_frame = drawDetections(frame, detections);
            out.write(vis_frame);
        }

        cap.release();
        out.release();
        std::cout << "Saved output video to " << args.out_file << std::endl;
    }

private:
    std::vector<std::string> class_names_;
    std::map<std::string, cv::Scalar> colors_;

    std::string getClassName(int class_id) {
        if (class_id >= 0 && class_id < static_cast<int>(class_names_.size())) {
            return class_names_[class_id];
        }
        return "class_" + std::to_string(class_id);
    }

    cv::Scalar getColor(const std::string& class_name) {
        auto it = colors_.find(class_name);
        if (it != colors_.end()) {
            return it->second;
        }
        return cv::Scalar(0, 255, 0);
    }

    void drawTranslucentBox(cv::Mat& image, int x1, int y1, int x2, int y2,
                           const cv::Scalar& color, float alpha) {
        try {
            cv::Mat roi = image(cv::Rect(x1, y1, x2 - x1, y2 - y1));
            cv::Mat color_overlay(roi.size(), roi.type(), color);
            cv::addWeighted(roi, 1.0 - alpha, color_overlay, alpha, 0, roi);
        } catch (...) {
            
        }
    }

    void drawLabel(cv::Mat& image, int x1, int y1, int x2, int y2,
                   const std::string& class_name, float score,
                   const cv::Scalar& color, int img_w, int img_h) {
        int score_percent = static_cast<int>(score * 100);
        std::string label_text = class_name + " - " + std::to_string(score_percent) + "%";

        double font_scale = std::max(0.35, std::min(img_w, img_h) / 1000.0);
        int thickness = std::max(1, static_cast<int>(std::round(font_scale * 2)));
        int font = cv::FONT_HERSHEY_SIMPLEX;

        int baseline;
        cv::Size text_size = cv::getTextSize(label_text, font, font_scale, thickness, &baseline);

        int pad_x = 6, pad_y = 4;
        int rect_x1 = x1 - 1;
        int rect_x2 = x1 + text_size.width + pad_x;
        int rect_y1 = y1 - text_size.height - baseline - pad_y;
        int rect_y2 = y1;
        cv::Point text_pos(x1 + 3, y1 - 4);

        if (rect_y1 < 0) {
            rect_y1 = y1;
            rect_y2 = y1 + text_size.height + baseline + pad_y;
            text_pos = cv::Point(x1 + 3, rect_y1 + text_size.height + baseline - 3);
        }

        rect_x1 = std::max(0, std::min(rect_x1, img_w));
        rect_x2 = std::max(0, std::min(rect_x2, img_w));
        rect_y1 = std::max(0, std::min(rect_y1, img_h));
        rect_y2 = std::max(0, std::min(rect_y2, img_h));

        cv::rectangle(image, cv::Point(rect_x1, rect_y1), cv::Point(rect_x2, rect_y2),
                     color, cv::FILLED);
        cv::putText(image, label_text, text_pos, font, font_scale,
                   cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
    }
};

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n"
              << "Options:\n"
              << "  --media_link PATH       Path to image/video (required)\n"
              << "  --model_path PATH       Path to ONNX model (required)\n"
              << "  --class_names NAMES     Space-separated class names\n"
              << "  --nms_thr FLOAT         NMS threshold (default: 0.45)\n"
              << "  --score_thr FLOAT       Score threshold (default: 0.25)\n"
              << "  --show                  Show live detections\n"
              << "  --out_file PATH         Output file path (default: output.mp4)\n";
}

Args parseArgs(int argc, char* argv[]) {
    Args args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--media_link" && i + 1 < argc) {
            args.media_link = argv[++i];
        } else if (arg == "--model_path" && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if (arg == "--class_names") {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                args.class_names.push_back(argv[++i]);
            }
        } else if (arg == "--nms_thr" && i + 1 < argc) {
            args.nms_thr = std::stof(argv[++i]);
        } else if (arg == "--score_thr" && i + 1 < argc) {
            args.score_thr = std::stof(argv[++i]);
        } else if (arg == "--show") {
            args.show = true;
        } else if (arg == "--out_file" && i + 1 < argc) {
            args.out_file = argv[++i];
        }
    }

    return args;
}

bool isVideoFile(const std::string& path) {
    std::vector<std::string> video_exts = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"};
    for (const auto& ext : video_exts) {
        if (path.size() >= ext.size() &&
            path.compare(path.size() - ext.size(), ext.size(), ext) == 0) {
            return true;
        }
    }
    return false;
}

bool isRTSP(const std::string& path) {
    return path.substr(0, 7) == "rtsp://";
}

void processVideoStream(cv::VideoCapture& cap, YOLOXDetector& detector,
                       Visualizer& visualizer, const Args& args) {
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 25.0;

    if (args.show) {
        visualizer.processVideoLive(cap, detector, args);
    } else {
        visualizer.processVideoToFile(cap, detector, args, fps, width, height);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    try {
        Args args = parseArgs(argc, argv);

        if (args.media_link.empty() || args.model_path.empty()) {
            std::cerr << "Error: --media_link and --model_path are required\n";
            printUsage(argv[0]);
            return 1;
        }

        YOLOXDetector detector(args.model_path);
        Visualizer visualizer(args.class_names);

        if (isVideoFile(args.media_link) || isRTSP(args.media_link)) {
            cv::VideoCapture cap(args.media_link);
            if (!cap.isOpened()) {
                throw std::runtime_error("Cannot open video/stream: " + args.media_link);
            }
            processVideoStream(cap, detector, visualizer, args);
        } else {
            visualizer.processImage(args.media_link, detector, args);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}