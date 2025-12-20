#include "ModelSerializer.hpp"

#include <stdexcept>
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <unordered_map>

// Layers
#include "Layer.hpp"
#include "ConvolutionLayer.hpp"
#include "FullyConnectedLayer.hpp"
#include "BatchNormalizationLayer.hpp"
#include "AveragePoolingLayer.hpp"
#include "MaxPoolingLayer.hpp"
#include "FlattenLayer.hpp"
#include "ActivationLayer.hpp"

// ONNX protobuf
#include "onnx-ml.pb.h"
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace
{
    constexpr int64_t kDefaultN = 1;
    constexpr int64_t kDefaultC = 3;

    std::string s(const char *base, int idx)
    {
        return std::string(base) + std::to_string(idx);
    }

    // ----------------------- EXPORT HELPERS -----------------------

    onnx::TensorProto makeTensor1f(const std::string &name, const Eigen::Tensor<double, 1> &t)
    {
        onnx::TensorProto tp;
        tp.set_name(name);
        tp.set_data_type(onnx::TensorProto_DataType_FLOAT);
        tp.add_dims(static_cast<int64_t>(t.dimension(0)));

        tp.mutable_float_data()->Reserve(static_cast<int>(t.dimension(0)));
        for (int i = 0; i < t.dimension(0); ++i)
            tp.add_float_data(static_cast<float>(t(i)));

        return tp;
    }

    onnx::TensorProto makeTensor4f(const std::string &name, const Eigen::Tensor<double, 4> &t)
    {
        onnx::TensorProto tp;
        tp.set_name(name);
        tp.set_data_type(onnx::TensorProto_DataType_FLOAT);
        tp.add_dims(static_cast<int64_t>(t.dimension(0)));
        tp.add_dims(static_cast<int64_t>(t.dimension(1)));
        tp.add_dims(static_cast<int64_t>(t.dimension(2)));
        tp.add_dims(static_cast<int64_t>(t.dimension(3)));

        const int64_t d0 = t.dimension(0);
        const int64_t d1 = t.dimension(1);
        const int64_t d2 = t.dimension(2);
        const int64_t d3 = t.dimension(3);

        tp.mutable_float_data()->Reserve(static_cast<int>(d0 * d1 * d2 * d3));

        for (int64_t i0 = 0; i0 < d0; ++i0)
            for (int64_t i1 = 0; i1 < d1; ++i1)
                for (int64_t i2 = 0; i2 < d2; ++i2)
                    for (int64_t i3 = 0; i3 < d3; ++i3)
                        tp.add_float_data(static_cast<float>(t(static_cast<int>(i0),
                                                               static_cast<int>(i1),
                                                               static_cast<int>(i2),
                                                               static_cast<int>(i3))));
        return tp;
    }

    onnx::TensorProto makeFCWeight2f(const std::string &name, const Eigen::Tensor<double, 4> &w4)
    {
        const int64_t out = w4.dimension(0);
        const int64_t in = w4.dimension(3);

        onnx::TensorProto tp;
        tp.set_name(name);
        tp.set_data_type(onnx::TensorProto_DataType_FLOAT);
        tp.add_dims(out);
        tp.add_dims(in);
        tp.mutable_float_data()->Reserve(static_cast<int>(out * in));

        for (int64_t o = 0; o < out; ++o)
            for (int64_t i = 0; i < in; ++i)
                tp.add_float_data(static_cast<float>(w4(static_cast<int>(o), 0, 0, static_cast<int>(i))));

        return tp;
    }

    void addModelInputDynamicHW(onnx::GraphProto &g, const std::string &name, int64_t n, int64_t c)
    {
        onnx::ValueInfoProto *v = g.add_input();
        v->set_name(name);

        auto *tt = v->mutable_type()->mutable_tensor_type();
        tt->set_elem_type(onnx::TensorProto_DataType_FLOAT);

        auto *shape = tt->mutable_shape();
        shape->add_dim()->set_dim_value(n);
        shape->add_dim()->set_dim_value(c);
        shape->add_dim()->set_dim_param("H");
        shape->add_dim()->set_dim_param("W");
    }

    void addModelOutput(onnx::GraphProto &g,
                        const std::string &name)
    {
        onnx::ValueInfoProto *v = g.add_output();
        v->set_name(name);
        v->mutable_type()->mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_FLOAT);
    }

    void writeModelToFile(const onnx::ModelProto &model, const std::string &path)
    {
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs)
            throw std::runtime_error("Failed to open ONNX output file: " + path);

        google::protobuf::io::OstreamOutputStream out(&ofs);
        if (!model.SerializeToZeroCopyStream(&out))
            throw std::runtime_error("Failed to serialize ONNX model: " + path);
    }

    std::string jsonEscape(const std::string &in)
    {
        std::string out;
        out.reserve(in.size());
        for (char c : in)
        {
            if (c == '\\')
                out += "\\\\";
            else if (c == '"')
                out += "\\\"";
            else
                out += c;
        }
        return out;
    }

    void writeClassMappingJsonNextToOnnx(const std::string &onnxPath,
                                         const std::vector<std::string> &classNames)
    {
        if (classNames.empty())
            return;

        namespace fs = std::filesystem;
        const fs::path jsonP = fs::path(onnxPath).parent_path() / "class_mapping.json";

        std::ofstream ofs(jsonP.string());
        if (!ofs)
            throw std::runtime_error("Failed to open: " + jsonP.string());

        ofs << "{\n  \"name_to_index\": {\n";
        for (size_t i = 0; i < classNames.size(); ++i)
        {
            ofs << "    \"" << jsonEscape(classNames[i]) << "\": " << i;
            ofs << (i + 1 < classNames.size() ? ",\n" : "\n");
        }
        ofs << "  }\n}\n";
    }

    void addTransposeNCHW_to_NWHC(onnx::GraphProto &g,
                                  const std::string &inName,
                                  const std::string &outName,
                                  int &nodeIdx)
    {
        onnx::NodeProto *n = g.add_node();
        n->set_op_type("Transpose");
        n->set_name(s("TransposeNode_", nodeIdx++));
        n->add_input(inName);
        n->add_output(outName);

        auto *perm = n->add_attribute();
        perm->set_name("perm");
        perm->set_type(onnx::AttributeProto_AttributeType_INTS);
        perm->add_ints(0);
        perm->add_ints(3);
        perm->add_ints(2);
        perm->add_ints(1);
    }

    // ----------------------- IMPORT HELPERS -----------------------

    void readModelFromFile(onnx::ModelProto &model, const std::string &path)
    {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs)
            throw std::runtime_error("Failed to open ONNX file for import: " + path);

        google::protobuf::io::IstreamInputStream in(&ifs);
        if (!model.ParseFromZeroCopyStream(&in))
            throw std::runtime_error("Failed to parse ONNX model: " + path);
    }

    std::unordered_map<std::string, onnx::TensorProto> buildInitializerMap(const onnx::GraphProto &g)
    {
        std::unordered_map<std::string, onnx::TensorProto> m;
        m.reserve(static_cast<size_t>(g.initializer_size()));
        for (const auto &tp : g.initializer())
            m.emplace(tp.name(), tp);
        return m;
    }

    const onnx::TensorProto &getInitOrThrow(const std::unordered_map<std::string, onnx::TensorProto> &m,
                                            const std::string &name)
    {
        auto it = m.find(name);
        if (it == m.end())
            throw std::runtime_error("importOnnx: missing initializer: " + name);
        return it->second;
    }

    std::vector<int64_t> dimsOf(const onnx::TensorProto &tp)
    {
        std::vector<int64_t> d;
        d.reserve(static_cast<size_t>(tp.dims_size()));
        for (int i = 0; i < tp.dims_size(); ++i)
            d.push_back(tp.dims(i));
        return d;
    }

    // Your exporter writes float_data, so we read float_data primarily.
    inline float fAt(const onnx::TensorProto &tp, int idx)
    {
        return tp.float_data(idx);
    }

    Eigen::Tensor<double, 1> tensor1_from_tp(const onnx::TensorProto &tp)
    {
        auto d = dimsOf(tp);
        if (d.size() != 1)
            throw std::runtime_error("importOnnx: expected 1D tensor initializer: " + tp.name());

        const int n = static_cast<int>(d[0]);
        if (tp.float_data_size() != n)
            throw std::runtime_error("importOnnx: bad float_data_size for 1D tensor: " + tp.name());

        Eigen::Tensor<double, 1> t(n);
        for (int i = 0; i < n; ++i)
            t(i) = static_cast<double>(fAt(tp, i));
        return t;
    }

    // IMPORTANT: recreate in the exact nested-loop order used in makeTensor4f.
    Eigen::Tensor<double, 4> tensor4_from_tp(const onnx::TensorProto &tp)
    {
        auto d = dimsOf(tp);
        if (d.size() != 4)
            throw std::runtime_error("importOnnx: expected 4D tensor initializer: " + tp.name());

        const int d0 = static_cast<int>(d[0]);
        const int d1 = static_cast<int>(d[1]);
        const int d2 = static_cast<int>(d[2]);
        const int d3 = static_cast<int>(d[3]);

        const int64_t total = static_cast<int64_t>(d0) * d1 * d2 * d3;
        if (tp.float_data_size() != total)
            throw std::runtime_error("importOnnx: bad float_data_size for 4D tensor: " + tp.name());

        Eigen::Tensor<double, 4> t(d0, d1, d2, d3);

        int idx = 0;
        for (int i0 = 0; i0 < d0; ++i0)
            for (int i1 = 0; i1 < d1; ++i1)
                for (int i2 = 0; i2 < d2; ++i2)
                    for (int i3 = 0; i3 < d3; ++i3, ++idx)
                        t(i0, i1, i2, i3) = static_cast<double>(fAt(tp, idx));

        return t;
    }

    // FC weights are stored in ONNX as 2D [out,in] by makeFCWeight2f.
    Eigen::Tensor<double, 4> fc_w4_from_tp_2d(const onnx::TensorProto &tp)
    {
        auto d = dimsOf(tp);
        if (d.size() != 2)
            throw std::runtime_error("importOnnx: expected 2D FC weight tensor: " + tp.name());

        const int out = static_cast<int>(d[0]);
        const int in = static_cast<int>(d[1]);

        const int64_t total = static_cast<int64_t>(out) * in;
        if (tp.float_data_size() != total)
            throw std::runtime_error("importOnnx: bad float_data_size for FC 2D weight: " + tp.name());

        Eigen::Tensor<double, 4> w4(out, 1, 1, in);

        int idx = 0;
        for (int o = 0; o < out; ++o)
            for (int i = 0; i < in; ++i, ++idx)
                w4(o, 0, 0, i) = static_cast<double>(fAt(tp, idx));

        return w4;
    }

    std::vector<int> getIntsAttr(const onnx::NodeProto &n, const std::string &name)
    {
        for (const auto &a : n.attribute())
        {
            if (a.name() == name && a.type() == onnx::AttributeProto_AttributeType_INTS)
            {
                std::vector<int> out;
                out.reserve(static_cast<size_t>(a.ints_size()));
                for (int i = 0; i < a.ints_size(); ++i)
                    out.push_back(static_cast<int>(a.ints(i)));
                return out;
            }
        }
        return {};
    }

    float getFloatAttr(const onnx::NodeProto &n, const std::string &name, float def)
    {
        for (const auto &a : n.attribute())
        {
            if (a.name() == name && a.type() == onnx::AttributeProto_AttributeType_FLOAT)
                return a.f();
        }
        return def;
    }

    int getIntAttr(const onnx::NodeProto &n, const std::string &name, int def)
    {
        for (const auto &a : n.attribute())
        {
            if (a.name() == name && a.type() == onnx::AttributeProto_AttributeType_INT)
                return static_cast<int>(a.i());
        }
        return def;
    }

    bool isExportTransposeNCHW_to_NWHC(const onnx::NodeProto &n)
    {
        if (n.op_type() != "Transpose")
            return false;
        auto perm = getIntsAttr(n, "perm");
        return (perm.size() == 4 && perm[0] == 0 && perm[1] == 3 && perm[2] == 2 && perm[3] == 1);
    }

    std::vector<std::string> readClassMappingJsonNextToOnnx(const std::string &onnxPath)
    {
        namespace fs = std::filesystem;
        const fs::path jsonP = fs::path(onnxPath).parent_path() / "class_mapping.json";

        std::vector<std::string> out;
        if (!fs::exists(jsonP))
            return out; // optional file

        std::ifstream ifs(jsonP.string());
        if (!ifs)
            throw std::runtime_error("Failed to open: " + jsonP.string());

        std::string s;
        s.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());

        struct Pair
        {
            std::string name;
            int index;
        };
        std::vector<Pair> pairs;

        size_t i = 0;
        while (i < s.size())
        {
            while (i < s.size() && s[i] != '"')
                ++i;
            if (i >= s.size())
                break;

            size_t j = i + 1;
            while (j < s.size() && s[j] != '"')
                ++j;
            if (j >= s.size())
                break;

            std::string name = s.substr(i + 1, j - (i + 1));
            i = j + 1;

            while (i < s.size() && s[i] != ':')
                ++i;
            if (i >= s.size())
                break;
            ++i;

            while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
                ++i;

            bool neg = false;
            if (i < s.size() && s[i] == '-')
            {
                neg = true;
                ++i;
            }

            if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i])))
            {
                continue;
            }

            int val = 0;
            while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
            {
                val = val * 10 + (s[i] - '0');
                ++i;
            }
            if (neg)
                val = -val;

            if (val >= 0)
                pairs.push_back({name, val});
        }

        int maxIdx = -1;
        for (const auto &p : pairs)
            if (p.index > maxIdx)
                maxIdx = p.index;

        if (maxIdx < 0)
            return {};

        out.assign(static_cast<size_t>(maxIdx + 1), std::string{});
        for (const auto &p : pairs)
        {
            if (p.index >= 0 && p.index < static_cast<int>(out.size()))
                out[static_cast<size_t>(p.index)] = p.name;
        }

        for (size_t k = 0; k < out.size(); ++k)
        {
            if (out[k].empty())
                throw std::runtime_error("class_mapping.json has missing index: " + std::to_string(k));
        }

        return out;
    }

} // namespace

// ======================= EXPORT =======================

void ModelSerializer::exportOnnx(const std::string &onnxPath,
                                 const std::vector<std::shared_ptr<Layer>> &layers,
                                 const std::vector<std::string> &classNames)
{
    if (layers.empty())
        throw std::runtime_error("exportOnnx: layers is empty.");

    onnx::ModelProto model;
    model.set_ir_version(11);

    auto *opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(11);

    onnx::GraphProto *g = model.mutable_graph();
    g->set_name("cnn_cpp_export");

    const std::string inputName = "input";
    addModelInputDynamicHW(*g, inputName, kDefaultN, kDefaultC);

    std::string current = inputName;
    bool currentIs2D = false;

    int convIdx = 0, bnIdx = 0, poolIdx = 0, actIdx = 0, fcIdx = 0, flatIdx = 0, nodeIdx = 0;

    for (const auto &layer : layers)
    {
        if (auto conv = std::dynamic_pointer_cast<ConvolutionLayer>(layer))
        {
            const std::string wName = s("convW_", convIdx);
            const std::string bName = s("convB_", convIdx);
            const std::string outName = s("convOut_", convIdx);

            g->add_initializer()->CopyFrom(makeTensor4f(wName, conv->getKernels()));
            g->add_initializer()->CopyFrom(makeTensor1f(bName, conv->getBiases()));

            onnx::NodeProto *n = g->add_node();
            n->set_op_type("Conv");
            n->set_name(s("ConvNode_", nodeIdx++));
            n->add_input(current);
            n->add_input(wName);
            n->add_input(bName);
            n->add_output(outName);

            auto *strides = n->add_attribute();
            strides->set_name("strides");
            strides->set_type(onnx::AttributeProto_AttributeType_INTS);
            strides->add_ints(conv->getStride());
            strides->add_ints(conv->getStride());

            const int p = conv->getPadding();
            auto *pads = n->add_attribute();
            pads->set_name("pads");
            pads->set_type(onnx::AttributeProto_AttributeType_INTS);
            pads->add_ints(p);
            pads->add_ints(p);
            pads->add_ints(p);
            pads->add_ints(p);

            const int k = conv->getKernelSize();
            auto *kshape = n->add_attribute();
            kshape->set_name("kernel_shape");
            kshape->set_type(onnx::AttributeProto_AttributeType_INTS);
            kshape->add_ints(k);
            kshape->add_ints(k);

            current = outName;
            currentIs2D = false;
            convIdx++;
            continue;
        }

        if (auto bn = std::dynamic_pointer_cast<BatchNormalizationLayer>(layer))
        {
            const std::string scaleName = s("bnScale_", bnIdx);
            const std::string biasName = s("bnBias_", bnIdx);
            const std::string meanName = s("bnMean_", bnIdx);
            const std::string varName = s("bnVar_", bnIdx);
            const std::string outName = s("bnOut_", bnIdx);

            Eigen::Tensor<double, 1> gamma = bn->getGamma();
            Eigen::Tensor<double, 1> beta = bn->getBeta();

            g->add_initializer()->CopyFrom(makeTensor1f(scaleName, gamma));
            g->add_initializer()->CopyFrom(makeTensor1f(biasName, beta));

            Eigen::Tensor<double, 1> mean(gamma.dimension(0));
            Eigen::Tensor<double, 1> var(gamma.dimension(0));
            mean.setZero();
            var.setConstant(1.0);

            g->add_initializer()->CopyFrom(makeTensor1f(meanName, mean));
            g->add_initializer()->CopyFrom(makeTensor1f(varName, var));

            onnx::NodeProto *n = g->add_node();
            n->set_op_type("BatchNormalization");
            n->set_name(s("BNNode_", nodeIdx++));
            n->add_input(current);
            n->add_input(scaleName);
            n->add_input(biasName);
            n->add_input(meanName);
            n->add_input(varName);
            n->add_output(outName);

            auto *eps = n->add_attribute();
            eps->set_name("epsilon");
            eps->set_type(onnx::AttributeProto_AttributeType_FLOAT);
            eps->set_f(1e-5f);

            auto *mom = n->add_attribute();
            mom->set_name("momentum");
            mom->set_type(onnx::AttributeProto_AttributeType_FLOAT);
            mom->set_f(0.9f);

            current = outName;
            bnIdx++;
            continue;
        }

        if (auto mp = std::dynamic_pointer_cast<MaxPoolingLayer>(layer))
        {
            const std::string outName = s("maxPoolOut_", poolIdx);

            onnx::NodeProto *n = g->add_node();
            n->set_op_type("MaxPool");
            n->set_name(s("MaxPoolNode_", nodeIdx++));
            n->add_input(current);
            n->add_output(outName);

            auto *kshape = n->add_attribute();
            kshape->set_name("kernel_shape");
            kshape->set_type(onnx::AttributeProto_AttributeType_INTS);
            kshape->add_ints(mp->getPoolSize());
            kshape->add_ints(mp->getPoolSize());

            auto *strides = n->add_attribute();
            strides->set_name("strides");
            strides->set_type(onnx::AttributeProto_AttributeType_INTS);
            strides->add_ints(mp->getStride());
            strides->add_ints(mp->getStride());

            auto *pads = n->add_attribute();
            pads->set_name("pads");
            pads->set_type(onnx::AttributeProto_AttributeType_INTS);
            pads->add_ints(0);
            pads->add_ints(0);
            pads->add_ints(0);
            pads->add_ints(0);

            current = outName;
            currentIs2D = false;
            poolIdx++;
            continue;
        }

        if (auto ap = std::dynamic_pointer_cast<AveragePoolingLayer>(layer))
        {
            const std::string outName = s("avgPoolOut_", poolIdx);

            onnx::NodeProto *n = g->add_node();
            n->set_op_type("AveragePool");
            n->set_name(s("AvgPoolNode_", nodeIdx++));
            n->add_input(current);
            n->add_output(outName);

            auto *kshape = n->add_attribute();
            kshape->set_name("kernel_shape");
            kshape->set_type(onnx::AttributeProto_AttributeType_INTS);
            kshape->add_ints(ap->getPoolSize());
            kshape->add_ints(ap->getPoolSize());

            auto *strides = n->add_attribute();
            strides->set_name("strides");
            strides->set_type(onnx::AttributeProto_AttributeType_INTS);
            strides->add_ints(ap->getStride());
            strides->add_ints(ap->getStride());

            auto *pads = n->add_attribute();
            pads->set_name("pads");
            pads->set_type(onnx::AttributeProto_AttributeType_INTS);
            pads->add_ints(0);
            pads->add_ints(0);
            pads->add_ints(0);
            pads->add_ints(0);

            current = outName;
            currentIs2D = false;
            poolIdx++;
            continue;
        }

        if (std::dynamic_pointer_cast<FlattenLayer>(layer))
        {
            const std::string trOut = s("transOut_", flatIdx);
            addTransposeNCHW_to_NWHC(*g, current, trOut, nodeIdx);

            const std::string outName = s("flatOut_", flatIdx);

            onnx::NodeProto *n = g->add_node();
            n->set_op_type("Flatten");
            n->set_name(s("FlattenNode_", nodeIdx++));
            n->add_input(trOut);
            n->add_output(outName);

            auto *axis = n->add_attribute();
            axis->set_name("axis");
            axis->set_type(onnx::AttributeProto_AttributeType_INT);
            axis->set_i(1);

            current = outName;
            currentIs2D = true;
            flatIdx++;
            continue;
        }

        if (auto act = std::dynamic_pointer_cast<ActivationLayer>(layer))
        {
            const std::string outName = s("actOut_", actIdx);

            onnx::NodeProto *n = g->add_node();
            n->set_name(s("ActNode_", nodeIdx++));
            n->add_input(current);
            n->add_output(outName);

            switch (act->getType())
            {
            case ActivationType::RELU:
                n->set_op_type("Relu");
                break;
            case ActivationType::LEAKY_RELU:
            {
                n->set_op_type("LeakyRelu");
                auto *a = n->add_attribute();
                a->set_name("alpha");
                a->set_type(onnx::AttributeProto_AttributeType_FLOAT);
                a->set_f(static_cast<float>(act->getAlpha()));
                break;
            }
            case ActivationType::SIGMOID:
                n->set_op_type("Sigmoid");
                break;
            case ActivationType::TANH:
                n->set_op_type("Tanh");
                break;
            case ActivationType::ELU:
            {
                n->set_op_type("Elu");
                auto *a = n->add_attribute();
                a->set_name("alpha");
                a->set_type(onnx::AttributeProto_AttributeType_FLOAT);
                a->set_f(static_cast<float>(act->getAlpha()));
                break;
            }
            case ActivationType::SOFTMAX:
            {
                n->set_op_type("Softmax");
                auto *axis = n->add_attribute();
                axis->set_name("axis");
                axis->set_type(onnx::AttributeProto_AttributeType_INT);
                axis->set_i(currentIs2D ? 1 : 3);
                break;
            }
            default:
                throw std::runtime_error("exportOnnx: unsupported activation.");
            }

            current = outName;
            actIdx++;
            continue;
        }

        if (auto fc = std::dynamic_pointer_cast<FullyConnectedLayer>(layer))
        {
            if (!currentIs2D)
            {
                const std::string trOut = s("transAuto_", flatIdx);
                addTransposeNCHW_to_NWHC(*g, current, trOut, nodeIdx);

                const std::string flatOut = s("flatAuto_", flatIdx);

                onnx::NodeProto *n = g->add_node();
                n->set_op_type("Flatten");
                n->set_name(s("FlattenAutoNode_", nodeIdx++));
                n->add_input(trOut);
                n->add_output(flatOut);

                auto *axis = n->add_attribute();
                axis->set_name("axis");
                axis->set_type(onnx::AttributeProto_AttributeType_INT);
                axis->set_i(1);

                current = flatOut;
                currentIs2D = true;
                flatIdx++;
            }

            const std::string wName = s("fcW_", fcIdx);
            const std::string bName = s("fcB_", fcIdx);
            const std::string outName = s("fcOut_", fcIdx);

            g->add_initializer()->CopyFrom(makeFCWeight2f(wName, fc->getWeights()));
            g->add_initializer()->CopyFrom(makeTensor1f(bName, fc->getBiases()));

            onnx::NodeProto *n = g->add_node();
            n->set_op_type("Gemm");
            n->set_name(s("GemmNode_", nodeIdx++));
            n->add_input(current);
            n->add_input(wName);
            n->add_input(bName);
            n->add_output(outName);

            auto *transB = n->add_attribute();
            transB->set_name("transB");
            transB->set_type(onnx::AttributeProto_AttributeType_INT);
            transB->set_i(1);

            current = outName;
            currentIs2D = true;
            fcIdx++;
            continue;
        }

        throw std::runtime_error("exportOnnx: unsupported layer type.");
    }

    addModelOutput(*g, current);

    model.set_producer_name("CNN-CPP Framework");
    model.set_producer_version("1.0");

    writeModelToFile(model, onnxPath);
    writeClassMappingJsonNextToOnnx(onnxPath, classNames);

    std::cout << "ONNX export finished: " << onnxPath << "\n";
}

void ModelSerializer::importOnnx(const std::string &onnxPath,
                                 std::vector<std::shared_ptr<Layer>> &layers,
                                 std::vector<std::string> &classNamesOut)
{
    layers.clear();
    classNamesOut.clear();

    onnx::ModelProto model;
    readModelFromFile(model, onnxPath);

    const onnx::GraphProto &g = model.graph();
    const auto inits = buildInitializerMap(g);

    bool currentIs2D = false;

    for (int ni = 0; ni < g.node_size(); ++ni)
    {
        const onnx::NodeProto &n = g.node(ni);

        if (isExportTransposeNCHW_to_NWHC(n))
            continue;

        if (n.op_type() == "Conv")
        {
            if (n.input_size() < 3)
                throw std::runtime_error("importOnnx: Conv node missing inputs.");

            const auto &wTp = getInitOrThrow(inits, n.input(1));
            const auto &bTp = getInitOrThrow(inits, n.input(2));

            // Reconstruct tensors
            Eigen::Tensor<double, 4> kernels = tensor4_from_tp(wTp);
            Eigen::Tensor<double, 1> biases = tensor1_from_tp(bTp);

            // Attributes
            const auto strides = getIntsAttr(n, "strides");
            const int stride = strides.empty() ? 1 : strides[0];

            const auto pads = getIntsAttr(n, "pads");
            const int padding = pads.empty() ? 0 : pads[0];

            const auto kshape = getIntsAttr(n, "kernel_shape");
            const int kernel_size = kshape.empty() ? static_cast<int>(kernels.dimension(2)) : kshape[0];

            auto conv = std::make_shared<ConvolutionLayer>(
                static_cast<int>(kernels.dimension(0)),
                kernel_size,
                stride,
                padding,
                ConvKernelInitialization::HE,
                ConvBiasInitialization::ZERO);

            conv->setInputDepth(static_cast<int>(kernels.dimension(1)), false);
            conv->setKernels(kernels);
            conv->setBiases(biases);

            layers.push_back(conv);
            currentIs2D = false;
            continue;
        }

        if (n.op_type() == "BatchNormalization")
        {
            // Inputs: [X, scale(gamma), bias(beta), mean, var]
            if (n.input_size() < 5)
                throw std::runtime_error("importOnnx: BatchNormalization node missing inputs.");

            const auto &scaleTp = getInitOrThrow(inits, n.input(1));
            const auto &biasTp = getInitOrThrow(inits, n.input(2));

            Eigen::Tensor<double, 1> gamma = tensor1_from_tp(scaleTp);
            Eigen::Tensor<double, 1> beta = tensor1_from_tp(biasTp);

            const double eps = static_cast<double>(getFloatAttr(n, "epsilon", 1e-5f));
            const double mom = static_cast<double>(getFloatAttr(n, "momentum", 0.9f));

            auto bn = std::make_shared<BatchNormalizationLayer>(eps, mom);

            if (layers.empty())
                throw std::runtime_error("importOnnx: BN cannot be first layer.");

            if (dynamic_cast<ConvolutionLayer *>(layers.back().get()))
                bn->setTarget(BNTarget::ConvolutionLayer);
            else if (dynamic_cast<FullyConnectedLayer *>(layers.back().get()))
                bn->setTarget(BNTarget::DenseLayer);
            else
                throw std::runtime_error("importOnnx: BN after unsupported layer type.");

            bn->setGamma(gamma);
            bn->setBeta(beta);

            layers.push_back(bn);
            continue;
        }

        if (n.op_type() == "MaxPool")
        {
            const auto kshape = getIntsAttr(n, "kernel_shape");
            const auto strides = getIntsAttr(n, "strides");

            const int pool_size = kshape.empty() ? 2 : kshape[0];
            const int stride = strides.empty() ? pool_size : strides[0];

            layers.push_back(std::make_shared<MaxPoolingLayer>(pool_size, stride));
            currentIs2D = false;
            continue;
        }

        if (n.op_type() == "AveragePool")
        {
            const auto kshape = getIntsAttr(n, "kernel_shape");
            const auto strides = getIntsAttr(n, "strides");

            const int pool_size = kshape.empty() ? 2 : kshape[0];
            const int stride = strides.empty() ? pool_size : strides[0];

            layers.push_back(std::make_shared<AveragePoolingLayer>(pool_size, stride));
            currentIs2D = false;
            continue;
        }

        if (n.op_type() == "Flatten")
        {
            layers.push_back(std::make_shared<FlattenLayer>());
            currentIs2D = true;
            continue;
        }

        // Activations
        if (n.op_type() == "Relu")
        {
            layers.push_back(std::make_shared<ActivationLayer>(ActivationType::RELU));
            continue;
        }
        if (n.op_type() == "LeakyRelu")
        {
            auto act = std::make_shared<ActivationLayer>(ActivationType::LEAKY_RELU);
            layers.push_back(act);
            continue;
        }
        if (n.op_type() == "Sigmoid")
        {
            layers.push_back(std::make_shared<ActivationLayer>(ActivationType::SIGMOID));
            continue;
        }
        if (n.op_type() == "Tanh")
        {
            layers.push_back(std::make_shared<ActivationLayer>(ActivationType::TANH));
            continue;
        }
        if (n.op_type() == "Elu")
        {
            auto act = std::make_shared<ActivationLayer>(ActivationType::ELU);
            layers.push_back(act);
            continue;
        }
        if (n.op_type() == "Softmax")
        {
            layers.push_back(std::make_shared<ActivationLayer>(ActivationType::SOFTMAX));
            continue;
        }

        if (n.op_type() == "Gemm")
        {
            // Inputs: [A, W, B]
            if (n.input_size() < 3)
                throw std::runtime_error("importOnnx: Gemm node missing inputs.");

            const auto &wTp = getInitOrThrow(inits, n.input(1));
            const auto &bTp = getInitOrThrow(inits, n.input(2));

            const int transB = getIntAttr(n, "transB", 1);
            if (transB != 1)
                throw std::runtime_error("importOnnx: Gemm transB != 1 not supported for CNN-CPP export.");

            Eigen::Tensor<double, 4> w4 = fc_w4_from_tp_2d(wTp);
            Eigen::Tensor<double, 1> b1 = tensor1_from_tp(bTp);

            const int outFeatures = static_cast<int>(w4.dimension(0));
            const int inFeatures = static_cast<int>(w4.dimension(3));

            auto fc = std::make_shared<FullyConnectedLayer>(
                outFeatures,
                DenseWeightInitialization::XAVIER,
                DenseBiasInitialization::ZERO);

            fc->setInputSize(inFeatures, false);

            fc->setWeights(w4);
            fc->setBiases(b1);

            layers.push_back(fc);
            currentIs2D = true;
            continue;
        }

        throw std::runtime_error("importOnnx: unsupported op_type encountered: " + n.op_type());
    }

    if (layers.empty())
        throw std::runtime_error("importOnnx: produced empty layer list (bad/unsupported model?).");

    classNamesOut = readClassMappingJsonNextToOnnx(onnxPath);

    std::cout << "ONNX import finished: " << onnxPath << "\n";
    if (!classNamesOut.empty())
        std::cout << "Loaded class mapping (" << classNamesOut.size() << " classes) from class_mapping.json\n";
    else
        std::cout << "No class_mapping.json found next to model (will fall back to dataset label order).\n";

    std::cout << "ONNX import finished: " << onnxPath << "\n";
}
