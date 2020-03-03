/*
  MIT License

  Copyright (c) 2016-2020 HolyWu

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/

#include <cmath>
#include <string>

#include <VapourSynth.h>
#include <VSHelper.h>

#include <caffe/caffe.hpp>

#include "waifu2x.h"

struct Waifu2xData {
    VSNodeRef * node;
    VSVideoInfo vi;
    int scale, blockWidth, blockHeight, batch;
    bool tta;
    float * srcInterleaved, * dstInterleaved, * buffer;
    Waifu2x * waifu2x;
};

static bool isPowerOf2(const int i) noexcept {
    return i && !(i & (i - 1));
}

static Waifu2x::eWaifu2xError filter(const VSFrameRef * src, VSFrameRef * dst, Waifu2xData * const VS_RESTRICT d, const VSAPI * vsapi) noexcept {
    if (d->vi.format->colorFamily == cmRGB) {
        const int width = vsapi->getFrameWidth(src, 0);
        const int height = vsapi->getFrameHeight(src, 0);
        const int srcStride = vsapi->getStride(src, 0) / sizeof(float);
        const int dstStride = vsapi->getStride(dst, 0) / sizeof(float);
        const float * srcpR = reinterpret_cast<const float *>(vsapi->getReadPtr(src, 0));
        const float * srcpG = reinterpret_cast<const float *>(vsapi->getReadPtr(src, 1));
        const float * srcpB = reinterpret_cast<const float *>(vsapi->getReadPtr(src, 2));
        float * VS_RESTRICT dstpR = reinterpret_cast<float *>(vsapi->getWritePtr(dst, 0));
        float * VS_RESTRICT dstpG = reinterpret_cast<float *>(vsapi->getWritePtr(dst, 1));
        float * VS_RESTRICT dstpB = reinterpret_cast<float *>(vsapi->getWritePtr(dst, 2));

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                const unsigned pos = (width * y + x) * 3;
                d->srcInterleaved[pos + 0] = srcpR[x];
                d->srcInterleaved[pos + 1] = srcpG[x];
                d->srcInterleaved[pos + 2] = srcpB[x];
            }

            srcpR += srcStride;
            srcpG += srcStride;
            srcpB += srcStride;
        }

        const auto waifu2xError = d->waifu2x->waifu2x(d->scale, d->srcInterleaved, d->dstInterleaved, width, height, 3, width * 3 * sizeof(float), 3, d->vi.width * 3 * sizeof(float),
                                                      d->blockWidth, d->blockHeight, d->tta, d->batch);
        if (waifu2xError != Waifu2x::eWaifu2xError_OK)
            return waifu2xError;

        for (int y = 0; y < d->vi.height; y++) {
            for (int x = 0; x < d->vi.width; x++) {
                const unsigned pos = (d->vi.width * y + x) * 3;
                dstpR[x] = d->dstInterleaved[pos + 0];
                dstpG[x] = d->dstInterleaved[pos + 1];
                dstpB[x] = d->dstInterleaved[pos + 2];
            }

            dstpR += dstStride;
            dstpG += dstStride;
            dstpB += dstStride;
        }
    } else {
        for (int plane = 0; plane < d->vi.format->numPlanes; plane++) {
            const int srcWidth = vsapi->getFrameWidth(src, plane);
            const int dstWidth = vsapi->getFrameWidth(dst, plane);
            const int srcHeight = vsapi->getFrameHeight(src, plane);
            const int dstHeight = vsapi->getFrameHeight(dst, plane);
            const int srcStride = vsapi->getStride(src, plane) / sizeof(float);
            const int dstStride = vsapi->getStride(dst, plane) / sizeof(float);
            const float * srcp = reinterpret_cast<const float *>(vsapi->getReadPtr(src, plane));
            float * VS_RESTRICT dstp = reinterpret_cast<float *>(vsapi->getWritePtr(dst, plane));

            Waifu2x::eWaifu2xError waifu2xError;

            if (plane == 0) {
                waifu2xError = d->waifu2x->waifu2x(d->scale, srcp, dstp, srcWidth, srcHeight, 1, vsapi->getStride(src, plane), 1, vsapi->getStride(dst, plane),
                                                   d->blockWidth, d->blockHeight, d->tta, d->batch);
            } else {
                const float * input = srcp;
                float * VS_RESTRICT output = d->buffer;

                for (int y = 0; y < srcHeight; y++) {
                    for (int x = 0; x < srcWidth; x++)
                        output[x] = input[x] + 0.5f;

                    input += srcStride;
                    output += srcWidth;
                }

                waifu2xError = d->waifu2x->waifu2x(d->scale, d->buffer, dstp, srcWidth, srcHeight, 1, srcWidth * sizeof(float), 1, vsapi->getStride(dst, plane),
                                                   d->blockWidth, d->blockHeight, d->tta, d->batch);

                for (int y = 0; y < dstHeight; y++) {
                    for (int x = 0; x < dstWidth; x++)
                        dstp[x] -= 0.5f;

                    dstp += dstStride;
                }
            }

            if (waifu2xError != Waifu2x::eWaifu2xError_OK)
                return waifu2xError;
        }
    }

    return Waifu2x::eWaifu2xError_OK;
}

static void VS_CC waifu2xInit(VSMap * in, VSMap * out, void ** instanceData, VSNode * node, VSCore * core, const VSAPI * vsapi) {
    Waifu2xData * d = static_cast<Waifu2xData *>(*instanceData);
    vsapi->setVideoInfo(&d->vi, 1, node);
}

static const VSFrameRef * VS_CC waifu2xGetFrame(int n, int activationReason, void ** instanceData, void ** frameData, VSFrameContext * frameCtx, VSCore * core, const VSAPI * vsapi) {
    Waifu2xData * d = static_cast<Waifu2xData *>(*instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        caffe::Caffe::set_mode(caffe::Caffe::GPU);

        const VSFrameRef * src = vsapi->getFrameFilter(n, d->node, frameCtx);
        VSFrameRef * dst = vsapi->newVideoFrame(d->vi.format, d->vi.width, d->vi.height, src, core);

        const auto waifu2xError = filter(src, dst, d, vsapi);
        if (waifu2xError != Waifu2x::eWaifu2xError_OK) {
            const char * error;

            if (waifu2xError == Waifu2x::eWaifu2xError_InvalidParameter)
                error = "invalid parameter";
            else if (waifu2xError == Waifu2x::eWaifu2xError_FailedProcessCaffe)
                error = "failed process caffe";
            else
                error = "unknown error";

            vsapi->setFilterError((std::string{ "Waifu2x-caffe: " } + error).c_str(), frameCtx);
            vsapi->freeFrame(src);
            vsapi->freeFrame(dst);
            return nullptr;
        }

        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

static void VS_CC waifu2xFree(void * instanceData, VSCore * core, const VSAPI * vsapi) {
    Waifu2xData * d = static_cast<Waifu2xData *>(instanceData);

    vsapi->freeNode(d->node);

    delete[] d->srcInterleaved;
    delete[] d->dstInterleaved;
    delete[] d->buffer;

    Waifu2x::quit_liblary();

    delete d->waifu2x;
    delete d;
}

static void VS_CC waifu2xCreate(const VSMap * in, VSMap * out, void * userData, VSCore * core, const VSAPI * vsapi) {
    char * argv[] = { const_cast<char *>("") };
    Waifu2x::init_liblary(1, argv);

    google::SetLogDestination(google::GLOG_INFO, "");
    google::SetLogDestination(google::GLOG_WARNING, "");
    google::SetLogDestination(google::GLOG_ERROR, "error_log_");
    google::SetLogDestination(google::GLOG_FATAL, "error_log_");

    Waifu2xData d{};
    int err;

    d.node = vsapi->propGetNode(in, "clip", 0, nullptr);
    d.vi = *vsapi->getVideoInfo(d.node);

    VSPlugin * fmtcPlugin;
    unsigned iterTimesTwiceScaling;

    try {
        if (!isConstantFormat(&d.vi) || d.vi.format->sampleType != stFloat || d.vi.format->bitsPerSample != 32)
            throw std::string{ "only constant format 32 bit float input supported" };

        const int noise = int64ToIntS(vsapi->propGetInt(in, "noise", 0, &err));

        d.scale = int64ToIntS(vsapi->propGetInt(in, "scale", 0, &err));
        if (err)
            d.scale = 2;

        d.blockWidth = int64ToIntS(vsapi->propGetInt(in, "block_w", 0, &err));
        if (err)
            d.blockWidth = 128;

        d.blockHeight = int64ToIntS(vsapi->propGetInt(in, "block_h", 0, &err));
        if (err)
            d.blockHeight = d.blockWidth;

        int model = int64ToIntS(vsapi->propGetInt(in, "model", 0, &err));
        if (err)
            model = 6;

        bool cudnn = !!vsapi->propGetInt(in, "cudnn", 0, &err);
        if (err)
            cudnn = true;

        const int processor = int64ToIntS(vsapi->propGetInt(in, "processor", 0, &err));

        d.tta = !!vsapi->propGetInt(in, "tta", 0, &err);

        d.batch = int64ToIntS(vsapi->propGetInt(in, "batch", 0, &err));
        if (err)
            d.batch = 1;

        if (noise == -1 && d.scale == 1) {
            vsapi->propSetNode(out, "clip", d.node, paReplace);
            vsapi->freeNode(d.node);
            return;
        }

        if (noise < -1 || noise > 3)
            throw std::string{ "noise must be -1, 0, 1, 2, or 3" };

        if (d.scale < 1 || !isPowerOf2(d.scale))
            throw std::string{ "scale must be greater than or equal to 1 and be a power of 2" };

        if (d.blockWidth < 1)
            throw std::string{ "block_w must be greater than or equal to 1" };

        if (d.blockHeight < 1)
            throw std::string{ "block_h must be greater than or equal to 1" };

        if (model < 0 || model > 6)
            throw std::string{ "model must be 0, 1, 2, 3, 4, 5, or 6" };

        if (model == 0 && noise == 0)
            throw std::string{ "anime_style_art model does not support noise reduction level 0" };

        if (model == 6 && ((d.blockWidth & 3) || (d.blockHeight & 3)))
            throw std::string{ "block size of cunet model must be divisible by 4" };

        if (processor < 0)
            throw std::string{ "processor must be greater than or equal to 0" };

        if (d.batch < 1)
            throw std::string{ "batch must be greater than or equal to 1" };

        fmtcPlugin = vsapi->getPluginById("fmtconv", core);
        if (d.scale != 1 && d.vi.format->subSamplingW && !fmtcPlugin)
            throw std::string{ "fmtconv plugin is required for correcting the horizontal chroma shift" };

        if (d.scale != 1) {
            d.vi.width *= d.scale;
            d.vi.height *= d.scale;
            iterTimesTwiceScaling = static_cast<unsigned>(std::log2(d.scale));
        }

        if (d.vi.format->colorFamily == cmRGB) {
            d.srcInterleaved = new (std::nothrow) float[vsapi->getVideoInfo(d.node)->width * vsapi->getVideoInfo(d.node)->height * 3];
            d.dstInterleaved = new (std::nothrow) float[d.vi.width * d.vi.height * 3];
            if (!d.srcInterleaved || !d.dstInterleaved)
                throw std::string{ "malloc failure (srcInterleaved/dstInterleaved)" };
        } else {
            d.buffer = new (std::nothrow) float[d.vi.width * d.vi.height];
            if (!d.buffer)
                throw std::string{ "malloc failure (buffer)" };
        }

        const auto modelType = (d.scale == 1) ? Waifu2x::eWaifu2xModelTypeNoise : (noise == -1 ? Waifu2x::eWaifu2xModelTypeScale : Waifu2x::eWaifu2xModelTypeNoiseScale);

        const std::string pluginPath{ vsapi->getPluginPath(vsapi->getPluginById("com.holywu.waifu2x-caffe", core)) };
        std::string modelPath{ pluginPath.substr(0, pluginPath.find_last_of('/')) };
        if (model == 0)
            modelPath += "/models/anime_style_art";
        else if (model == 1)
            modelPath += "/models/anime_style_art_rgb";
        else if (model == 2)
            modelPath += "/models/photo";
        else if (model == 3)
            modelPath += "/models/upconv_7_anime_style_art_rgb";
        else if (model == 4)
            modelPath += "/models/upconv_7_photo";
        else if (model == 5)
            modelPath += "/models/upresnet10";
        else
            modelPath += "/models/cunet";

        d.waifu2x = new Waifu2x{};

        const auto waifu2xError = d.waifu2x->Init(modelType, noise, modelPath, cudnn ? "cudnn" : "gpu", processor);
        if (waifu2xError != Waifu2x::eWaifu2xError_OK) {
            const char * error;

            if (waifu2xError == Waifu2x::eWaifu2xError_InvalidParameter)
                error = "invalid parameter";
            else if (waifu2xError == Waifu2x::eWaifu2xError_FailedOpenModelFile)
                error = "failed open model file";
            else if (waifu2xError == Waifu2x::eWaifu2xError_FailedParseModelFile)
                error = "failed parse model file";
            else if (waifu2xError == Waifu2x::eWaifu2xError_FailedConstructModel)
                error = "failed construct model";
            else if (waifu2xError == Waifu2x::eWaifu2xError_FailedCudaCheck)
                error = "failed CUDA check";
            else
                error = "unknown error";

            throw std::string{ error } + " at initialization";
        }
    } catch (const std::string & error) {
        vsapi->setError(out, ("Waifu2x-caffe: " + error).c_str());
        vsapi->freeNode(d.node);
        return;
    }

    Waifu2xData * data = new Waifu2xData{ d };

    vsapi->createFilter(in, out, "Waifu2x-caffe", waifu2xInit, waifu2xGetFrame, waifu2xFree, fmParallelRequests, 0, data, core);

    if (d.scale != 1 && d.vi.format->subSamplingW) {
        const double offset = 0.5 * (1 << d.vi.format->subSamplingW) - 0.5;
        double shift = 0.0;
        for (unsigned times = 0; times < iterTimesTwiceScaling; times++)
            shift = shift * 2.0 + offset;

        VSNodeRef * node = vsapi->propGetNode(out, "clip", 0, nullptr);
        vsapi->clearMap(out);
        VSMap * args = vsapi->createMap();
        vsapi->propSetNode(args, "clip", node, paReplace);
        vsapi->freeNode(node);
        vsapi->propSetFloat(args, "sx", shift, paReplace);
        vsapi->propSetFloat(args, "planes", 2, paReplace);
        vsapi->propSetFloat(args, "planes", 3, paAppend);
        vsapi->propSetFloat(args, "planes", 3, paAppend);

        VSMap * ret = vsapi->invoke(fmtcPlugin, "resample", args);
        if (vsapi->getError(ret)) {
            vsapi->setError(out, vsapi->getError(ret));
            vsapi->freeMap(args);
            vsapi->freeMap(ret);
            return;
        }

        node = vsapi->propGetNode(ret, "clip", 0, nullptr);
        vsapi->freeMap(args);
        vsapi->freeMap(ret);
        vsapi->propSetNode(out, "clip", node, paReplace);
        vsapi->freeNode(node);
    }
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin * plugin) {
    configFunc("com.holywu.waifu2x-caffe", "caffe", "Image Super-Resolution using Deep Convolutional Neural Networks", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("Waifu2x",
                 "clip:clip;"
                 "noise:int:opt;"
                 "scale:int:opt;"
                 "block_w:int:opt;"
                 "block_h:int:opt;"
                 "model:int:opt;"
                 "cudnn:int:opt;"
                 "processor:int:opt;"
                 "tta:int:opt;"
                 "batch:int:opt;",
                 waifu2xCreate, nullptr, plugin);
}
