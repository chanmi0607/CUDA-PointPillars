#include "tensorrt.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <functional>
#include <cstring>

#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "check.hpp"

namespace TensorRT {

static class Logger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
      std::cerr << "[NVINFER LOG]: " << msg << std::endl;
    }
  }
} gLogger_;

static std::string format_shape(const nvinfer1::Dims &shape) {
  char buf[256] = {0};
  char *p = buf;
  for (int i = 0; i < shape.nbDims; ++i) {
    if (i + 1 < shape.nbDims) {
      p += sprintf(p, "%ld x ", static_cast<long>(shape.d[i]));
    } else {
      p += sprintf(p, "%ld", static_cast<long>(shape.d[i]));
    }
  }
  return buf;
}

static std::vector<unsigned char> load_file(const std::string &file) {
  std::ifstream in(file, std::ios::in | std::ios::binary);
  if (!in.is_open()) return {};

  in.seekg(0, std::ios::end);
  size_t length = static_cast<size_t>(in.tellg());

  std::vector<unsigned char> data;
  if (length > 0) {
    in.seekg(0, std::ios::beg);
    data.resize(length);
    in.read(reinterpret_cast<char *>(&data[0]), length);
  }
  in.close();
  return data;
}

static const char *data_type_string(nvinfer1::DataType dt) {
  switch (dt) {
    case nvinfer1::DataType::kFLOAT: return "Float32";
    case nvinfer1::DataType::kHALF:  return "Float16";
    case nvinfer1::DataType::kINT32: return "Int32";
    case nvinfer1::DataType::kINT8:  return "Int8";
    case nvinfer1::DataType::kBOOL:  return "Bool";
#if NV_TENSORRT_MAJOR >= 10
    case nvinfer1::DataType::kUINT8: return "UInt8";
#endif
    default: return "Unknown";
  }
}

template <typename T>
static void destroy_pointer(T *ptr) {
  if (ptr) delete ptr;
}

class __native_engine_context {
public:
  virtual ~__native_engine_context() { destroy(); }

  bool construct(const void *pdata, size_t size, const char *message_name) {
    destroy();

    if (pdata == nullptr || size == 0) {
      printf("Construct for empty data found.\n");
      return false;
    }

    runtime_ = std::shared_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(gLogger_), destroy_pointer<nvinfer1::IRuntime>);
    if (runtime_ == nullptr) {
      printf("Failed to create tensorRT runtime: %s.\n", message_name);
      return false;
    }

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(pdata, size),
        destroy_pointer<nvinfer1::ICudaEngine>);
    if (engine_ == nullptr) {
      printf("Failed to deserialize engine: %s\n", message_name);
      return false;
    }

    context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
        engine_->createExecutionContext(),
        destroy_pointer<nvinfer1::IExecutionContext>);
    if (context_ == nullptr) {
      printf("Failed to create execution context: %s\n", message_name);
      return false;
    }

    return context_ != nullptr;
  }

private:
  void destroy() {
    context_.reset();
    engine_.reset();
    runtime_.reset();
  }

public:
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
};

class EngineImplement : public Engine {
public:
  std::shared_ptr<__native_engine_context> context_;

  std::unordered_map<std::string, int> binding_name_to_index_;
  std::vector<std::string> index_to_binding_name_;
  std::vector<bool> is_input_flags_;

  virtual ~EngineImplement() = default;

  bool construct(const void *data, size_t size, const char *message_name) {
    context_ = std::make_shared<__native_engine_context>();
    if (!context_->construct(data, size, message_name)) {
      return false;
    }
    setup();
    return true;
  }

  bool load(const std::string &file) {
    auto data = load_file(file);
    if (data.empty()) {
      printf("An empty file has been loaded. Please confirm your file path: %s\n", file.c_str());
      return false;
    }
    return this->construct(data.data(), data.size(), file.c_str());
  }

  void setup() {
    auto engine = this->context_->engine_;
    int nbTensors = static_cast<int>(engine->getNbIOTensors());

    binding_name_to_index_.clear();
    index_to_binding_name_.clear();
    is_input_flags_.clear();

    index_to_binding_name_.reserve(nbTensors);
    is_input_flags_.reserve(nbTensors);

    for (int i = 0; i < nbTensors; ++i) {
      const char *tensorName = engine->getIOTensorName(i);
      std::string name = tensorName ? tensorName : "";
      binding_name_to_index_[name] = i;
      index_to_binding_name_.push_back(name);

      auto mode = engine->getTensorIOMode(tensorName);
      is_input_flags_.push_back(mode == nvinfer1::TensorIOMode::kINPUT);
    }
  }

  virtual int index(const std::string &name) override {
    auto iter = binding_name_to_index_.find(name);
    Assertf(iter != binding_name_to_index_.end(), "Can not found the binding name: %s", name.c_str());
    return iter->second;
  }

  virtual bool forward(const std::vector<const void *> &bindings, void *stream = nullptr, void *input_consum_event = nullptr) override {
    (void)input_consum_event;

    auto engine  = this->context_->engine_;
    auto context = this->context_->context_;

    int nbTensors = static_cast<int>(engine->getNbIOTensors());
    Assertf(static_cast<int>(bindings.size()) == nbTensors,
            "bindings.size() != number of IO tensors, got %d vs %d",
            static_cast<int>(bindings.size()), nbTensors);

    for (int i = 0; i < nbTensors; ++i) {
      const char *tensorName = engine->getIOTensorName(i);
      bool ok = context->setTensorAddress(tensorName, const_cast<void *>(bindings[i]));
      Assertf(ok, "Failed to set tensor address for: %s", tensorName);
    }

    return context->enqueueV3(reinterpret_cast<cudaStream_t>(stream));
  }

  virtual std::vector<int> run_dims(const std::string &name) override {
    return run_dims(index(name));
  }

  virtual std::vector<int> run_dims(int ibinding) override {
    const char *name = index_to_binding_name_.at(ibinding).c_str();
    auto dim = this->context_->context_->getTensorShape(name);
    return std::vector<int>(dim.d, dim.d + dim.nbDims);
  }

  virtual std::vector<int> static_dims(const std::string &name) override {
    return static_dims(index(name));
  }

  virtual std::vector<int> static_dims(int ibinding) override {
    const char *name = index_to_binding_name_.at(ibinding).c_str();
    auto dim = this->context_->engine_->getTensorShape(name);
    return std::vector<int>(dim.d, dim.d + dim.nbDims);
  }

  virtual int num_bindings() override {
    return static_cast<int>(this->context_->engine_->getNbIOTensors());
  }

  virtual bool is_input(int ibinding) override {
    return is_input_flags_.at(ibinding);
  }

  virtual bool set_run_dims(const std::string &name, const std::vector<int> &dims) override {
    return this->set_run_dims(index(name), dims);
  }

  virtual bool set_run_dims(int ibinding, const std::vector<int> &dims) override {
    const char *name = index_to_binding_name_.at(ibinding).c_str();

    if (!is_input_flags_.at(ibinding)) {
      return false;
    }

    nvinfer1::Dims d;
    d.nbDims = static_cast<int>(dims.size());
    for (int i = 0; i < d.nbDims; ++i) d.d[i] = dims[i];

    return this->context_->context_->setInputShape(name, d);
  }

  virtual int numel(const std::string &name) override {
    return numel(index(name));
  }

  virtual int numel(int ibinding) override {
    const char *name = index_to_binding_name_.at(ibinding).c_str();
    auto dim = this->context_->context_->getTensorShape(name);

    int out = 1;
    for (int i = 0; i < dim.nbDims; ++i) {
      if (dim.d[i] < 0) return -1;
      out *= dim.d[i];
    }
    return out;
  }

  virtual DType dtype(const std::string &name) override {
    return dtype(index(name));
  }

  virtual DType dtype(int ibinding) override {
    const char *name = index_to_binding_name_.at(ibinding).c_str();
    return static_cast<DType>(this->context_->engine_->getTensorDataType(name));
  }

  virtual bool has_dynamic_dim() override {
    int numTensors = static_cast<int>(this->context_->engine_->getNbIOTensors());
    for (int i = 0; i < numTensors; ++i) {
      const char *name = this->context_->engine_->getIOTensorName(i);
      nvinfer1::Dims dims = this->context_->engine_->getTensorShape(name);
      for (int j = 0; j < dims.nbDims; ++j) {
        if (dims.d[j] == -1) return true;
      }
    }
    return false;
  }

  virtual void print(const char *name = "TensorRT-Engine") override {
    printf("------------------------------------------------------\n");
    printf("%s is %s model\n", name, has_dynamic_dim() ? "Dynamic Shape" : "Static Shape");

    auto engine = this->context_->engine_;
    int numTensors = static_cast<int>(engine->getNbIOTensors());

    int num_input = 0;
    int num_output = 0;
    for (int i = 0; i < numTensors; ++i) {
      auto mode = engine->getTensorIOMode(engine->getIOTensorName(i));
      if (mode == nvinfer1::TensorIOMode::kINPUT) num_input++;
      else num_output++;
    }

    printf("Inputs: %d\n", num_input);
    int in_idx = 0;
    int out_idx = 0;
    for (int i = 0; i < numTensors; ++i) {
      const char *tensorName = engine->getIOTensorName(i);
      auto mode  = engine->getTensorIOMode(tensorName);
      auto dim   = engine->getTensorShape(tensorName);
      auto dtype = engine->getTensorDataType(tensorName);

      if (mode == nvinfer1::TensorIOMode::kINPUT) {
        printf("\t%d.%s : {%s} [%s]\n",
               in_idx++, tensorName, format_shape(dim).c_str(), data_type_string(dtype));
      }
    }

    printf("Outputs: %d\n", num_output);
    for (int i = 0; i < numTensors; ++i) {
      const char *tensorName = engine->getIOTensorName(i);
      auto mode  = engine->getTensorIOMode(tensorName);
      auto dim   = engine->getTensorShape(tensorName);
      auto dtype = engine->getTensorDataType(tensorName);

      if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
        printf("\t%d.%s : {%s} [%s]\n",
               out_idx++, tensorName, format_shape(dim).c_str(), data_type_string(dtype));
      }
    }

    printf("------------------------------------------------------\n");
  }
};

std::shared_ptr<Engine> load(const std::string &file) {
  std::shared_ptr<EngineImplement> impl(new EngineImplement());
  if (!impl->load(file)) impl.reset();
  return impl;
}

}  // namespace TensorRT