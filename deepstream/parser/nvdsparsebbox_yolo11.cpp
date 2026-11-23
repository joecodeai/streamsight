// Custom bounding-box parser for YOLOv8 / YOLO11 ONNX models in DeepStream nvinfer.
//
// The Ultralytics export has a single output of shape [batch, 4 + num_classes, num_anchors]
// (e.g. 1x84x8400 for COCO at 640x640). Rows 0..3 hold cx, cy, w, h in network-input
// pixels; the remaining rows hold per-class scores (no separate objectness score).
//
// nvinfer strips the batch axis, so this function sees [C, N]. Some exporters emit the
// transposed [N, C] layout; both are handled. NMS is left to nvinfer (cluster-mode=2).

#include <algorithm>
#include <iostream>
#include <vector>

#include "nvdsinfer_custom_impl.h"

namespace {

inline float clampf(float v, float lo, float hi) { return std::max(lo, std::min(v, hi)); }

}  // namespace

extern "C" bool NvDsInferParseYolo11(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                     NvDsInferNetworkInfo const &networkInfo,
                                     NvDsInferParseDetectionParams const &detectionParams,
                                     std::vector<NvDsInferParseObjectInfo> &objectList) {
    if (outputLayersInfo.empty()) {
        std::cerr << "NvDsInferParseYolo11: no output layers" << std::endl;
        return false;
    }
    const NvDsInferLayerInfo &layer = outputLayersInfo[0];
    if (layer.dataType != FLOAT) {
        std::cerr << "NvDsInferParseYolo11: expected FP32 output, got type " << layer.dataType << std::endl;
        return false;
    }
    const NvDsInferDims &dims = layer.inferDims;
    if (dims.numDims != 2) {
        std::cerr << "NvDsInferParseYolo11: expected a 2-D output (C x N), got " << dims.numDims << " dims"
                  << std::endl;
        return false;
    }

    // Channels-first unless the first axis is the larger one (anchors always outnumber channels).
    const bool channelsFirst = dims.d[0] < dims.d[1];
    const unsigned int channels = channelsFirst ? dims.d[0] : dims.d[1];
    const unsigned int anchors = channelsFirst ? dims.d[1] : dims.d[0];
    if (channels <= 4) return false;
    const unsigned int numClasses = channels - 4;

    const float *data = static_cast<const float *>(layer.buffer);
    auto at = [&](unsigned int c, unsigned int i) -> float {
        return channelsFirst ? data[c * anchors + i] : data[i * channels + c];
    };

    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);
    const unsigned int configured = detectionParams.numClassesConfigured;
    const auto &thresholds = detectionParams.perClassPreclusterThreshold;

    objectList.reserve(256);
    for (unsigned int i = 0; i < anchors; ++i) {
        unsigned int bestClass = 0;
        float bestScore = at(4, i);
        for (unsigned int c = 1; c < numClasses; ++c) {
            const float s = at(4 + c, i);
            if (s > bestScore) {
                bestScore = s;
                bestClass = c;
            }
        }
        if (bestClass >= configured) continue;
        if (bestScore < thresholds[bestClass]) continue;

        const float cx = at(0, i), cy = at(1, i), w = at(2, i), h = at(3, i);
        const float x1 = clampf(cx - w / 2.f, 0.f, netW);
        const float y1 = clampf(cy - h / 2.f, 0.f, netH);
        const float x2 = clampf(cx + w / 2.f, 0.f, netW);
        const float y2 = clampf(cy + h / 2.f, 0.f, netH);
        if (x2 - x1 < 1.f || y2 - y1 < 1.f) continue;

        NvDsInferParseObjectInfo obj;
        obj.classId = bestClass;
        obj.detectionConfidence = bestScore;
        obj.left = x1;
        obj.top = y1;
        obj.width = x2 - x1;
        obj.height = y2 - y1;
        objectList.push_back(obj);
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo11);
