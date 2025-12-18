#pragma once

#include <memory>
#include <string>
#include <vector>

class Layer;

class ModelSerializer
{
public:
    static void exportOnnx(const std::string &onnxPath,
                           const std::vector<std::shared_ptr<Layer>> &layers,
                           const std::vector<std::string> &classNames);

    static void importOnnx(const std::string &onnxPath,
                           std::vector<std::shared_ptr<Layer>> &layers,
                           std::vector<std::string> &classNamesOut);
};
