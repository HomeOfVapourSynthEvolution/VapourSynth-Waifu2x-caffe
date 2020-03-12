#pragma once

#include <caffe/common.hpp>
#include "caffe/layers/input_layer.hpp"
#include "caffe/layers/flatten_layer.hpp"
#include "caffe/layers/scale_layer.hpp"
#include "caffe/layers/crop_layer.hpp"

namespace caffe {
    extern INSTANTIATE_CLASS(InputLayer);
    extern INSTANTIATE_CLASS(FlattenLayer);
    extern INSTANTIATE_CLASS(ScaleLayer);
    extern INSTANTIATE_CLASS(CropLayer);
}
