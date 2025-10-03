#ifndef ONNX_SESSION_HPP
#define ONNX_SESSION_HPP

#include <onnxruntime_cxx_api.h>
#include <string>
#include <memory>

class ONNXSessionBuilder {
public:
    ONNXSessionBuilder(const std::string& model_path, int num_threads);
    ~ONNXSessionBuilder() = default;

    std::unique_ptr<Ort::Session> build();
    Ort::Env& getEnv() { return env_; }

private:
    std::string model_path_;
    int num_threads_;
    Ort::Env env_;

    OrtTensorRTProviderOptions getTensorRTOptions();
    OrtCUDAProviderOptions getCUDAOptions();
};

inline ONNXSessionBuilder::ONNXSessionBuilder(const std::string& model_path, int num_threads)
    : model_path_(model_path)
    , num_threads_(num_threads)
    , env_(ORT_LOGGING_LEVEL_WARNING, "ONNXSession")
{
}

inline OrtTensorRTProviderOptions ONNXSessionBuilder::getTensorRTOptions() {
    OrtTensorRTProviderOptions trtOptions{};
    trtOptions.device_id = 0;
    trtOptions.trt_max_workspace_size = 2147483648;
    trtOptions.trt_fp16_enable = 1;
    trtOptions.trt_engine_cache_enable = 1;
    trtOptions.trt_engine_cache_path = "./trt_cache";
    return trtOptions;
}

inline OrtCUDAProviderOptions ONNXSessionBuilder::getCUDAOptions() {
    OrtCUDAProviderOptions cudaOptions{};
    cudaOptions.device_id = 0;
    cudaOptions.gpu_mem_limit = 2ULL * 1024 * 1024 * 1024;
    cudaOptions.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
    cudaOptions.do_copy_in_default_stream = 1;
    return cudaOptions;
}

inline std::unique_ptr<Ort::Session> ONNXSessionBuilder::build() {
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(num_threads_);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    OrtTensorRTProviderOptions trtOptions = getTensorRTOptions();
    OrtCUDAProviderOptions cudaOptions = getCUDAOptions();

    // sessionOptions.AppendExecutionProvider_TensorRT(trtOptions);
    sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);

    return std::make_unique<Ort::Session>(env_, model_path_.c_str(), sessionOptions);
}

#endif // ONNX_SESSION_HPP