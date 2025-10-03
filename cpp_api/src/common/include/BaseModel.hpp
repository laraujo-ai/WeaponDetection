#ifndef INFERENCER_HPP
#define INFERENCER_HPP

#include "SessionBuilder.hpp"
#include <vector>
#include <iostream>

template<typename InputType, typename OutputType>
class IBaseModel {
public:
    IBaseModel(const std::string& model_path, int num_threads);
    virtual ~IBaseModel() = default;
    OutputType run(const InputType& input);

protected:
    virtual std::vector<Ort::Value> preprocess(const InputType& input) = 0;
    virtual std::vector<Ort::Value> infer(std::vector<Ort::Value>& input_tensors);
    virtual OutputType postprocess(std::vector<Ort::Value>& output_tensors) = 0;

    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

private:
    void extractModelMetadata();
};

template<typename InputType, typename OutputType>
IBaseModel<InputType, OutputType>::IBaseModel(const std::string& model_path, int num_threads)
    : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    ONNXSessionBuilder builder(model_path, num_threads);
    session_ = builder.build();
    extractModelMetadata();
}

template<typename InputType, typename OutputType>
OutputType IBaseModel<InputType, OutputType>::run(const InputType& input) {
    std::vector<Ort::Value> input_tensors = preprocess(input);
    std::vector<Ort::Value> output_tensors = infer(input_tensors);
    return postprocess(output_tensors);
}

template<typename InputType, typename OutputType>
std::vector<Ort::Value> IBaseModel<InputType, OutputType>::infer(std::vector<Ort::Value>& input_tensors) {
    std::vector<const char*> input_names_cstr;
    std::vector<const char*> output_names_cstr;

    for (const auto& name : input_names_) {
        input_names_cstr.push_back(name.c_str());
    }

    for (const auto& name : output_names_) {
        output_names_cstr.push_back(name.c_str());
    }

    return session_->Run(
        Ort::RunOptions{nullptr},
        input_names_cstr.data(),
        input_tensors.data(),
        input_tensors.size(),
        output_names_cstr.data(),
        output_names_cstr.size()
    );
}

template<typename InputType, typename OutputType>
void IBaseModel<InputType, OutputType>::extractModelMetadata() {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_inputs = session_->GetInputCount();

    for (size_t i = 0; i < num_inputs; ++i) {
        auto input_name = session_->GetInputNameAllocated(i, allocator);
        input_names_.push_back(input_name.get());
    }

    size_t num_outputs = session_->GetOutputCount();

    for (size_t i = 0; i < num_outputs; ++i) {
        auto output_name = session_->GetOutputNameAllocated(i, allocator);
        output_names_.push_back(output_name.get());
    }
}

#endif // INFERENCER_HPP