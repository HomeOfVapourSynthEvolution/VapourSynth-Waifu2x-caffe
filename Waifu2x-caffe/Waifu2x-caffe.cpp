/*
  The MIT License (MIT)

  Copyright (c) 2016 HolyWu

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
#include <caffe/caffe.hpp>
#include <vapoursynth/VapourSynth.h>
#include <vapoursynth/VSHelper.h>
#include "waifu2x.h"

struct Waifu2xData {
    VSNodeRef * node;
    VSVideoInfo vi;
    int scale;
    float * srcInterleaved, * dstInterleaved, * buffer;
    Waifu2x * waifu2x;
};

static inline bool isPowerOf2(const int i) {
    return i && !(i & (i - 1));
}

static Waifu2x::eWaifu2xError Process(const VSFrameRef * src, VSFrameRef * dst, Waifu2xData * VS_RESTRICT d, const VSAPI * vsapi) {
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
                const int pos = y * width + x;
                d->srcInterleaved[pos * 3] = srcpR[x];
                d->srcInterleaved[pos * 3 + 1] = srcpG[x];
                d->srcInterleaved[pos * 3 + 2] = srcpB[x];
            }
            srcpR += srcStride;
            srcpG += srcStride;
            srcpB += srcStride;
        }

        const Waifu2x::eWaifu2xError waifu2xError =
            d->waifu2x->waifu2x(d->scale, d->srcInterleaved, d->dstInterleaved, width, height, 3, width * 3 * sizeof(float), 3, d->vi.width * 3 * sizeof(float));
        if (waifu2xError != Waifu2x::eWaifu2xError_OK)
            return waifu2xError;

        for (int y = 0; y < d->vi.height; y++) {
            for (int x = 0; x < d->vi.width; x++) {
                const int pos = y * d->vi.width + x;
                dstpR[x] = d->dstInterleaved[pos * 3];
                dstpG[x] = d->dstInterleaved[pos * 3 + 1];
                dstpB[x] = d->dstInterleaved[pos * 3 + 2];
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
                waifu2xError = d->waifu2x->waifu2x(d->scale, srcp, dstp, srcWidth, srcHeight, 1, vsapi->getStride(src, plane), 1, vsapi->getStride(dst, plane));
            } else {
                const float * input = srcp;
                float * VS_RESTRICT output = d->buffer;

                for (int y = 0; y < srcHeight; y++) {
                    for (int x = 0; x < srcWidth; x++)
                        output[x] = input[x] + 0.5f;
                    input += srcStride;
                    output += srcWidth;
                }

                waifu2xError = d->waifu2x->waifu2x(d->scale, d->buffer, dstp, srcWidth, srcHeight, 1, srcWidth * sizeof(float), 1, vsapi->getStride(dst, plane));

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

static void VS_CC waifu2xInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    Waifu2xData * d = static_cast<Waifu2xData *>(*instanceData);
    vsapi->setVideoInfo(&d->vi, 1, node);
}

static const VSFrameRef *VS_CC waifu2xGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    Waifu2xData * d = static_cast<Waifu2xData *>(*instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        caffe::Caffe::set_mode(caffe::Caffe::GPU);

        const VSFrameRef * src = vsapi->getFrameFilter(n, d->node, frameCtx);
        VSFrameRef * dst = vsapi->newVideoFrame(d->vi.format, d->vi.width, d->vi.height, src, core);

        const Waifu2x::eWaifu2xError waifu2xError = Process(src, dst, d, vsapi);
        if (waifu2xError != Waifu2x::eWaifu2xError_OK) {
            const char * err;

            switch (waifu2xError) {
            case Waifu2x::eWaifu2xError_InvalidParameter:
                err = "invalid parameter";
                break;
            case Waifu2x::eWaifu2xError_FailedProcessCaffe:
                err = "failed process caffe";
                break;
            }

            vsapi->setFilterError((std::string("Waifu2x-caffe: ") + err).c_str(), frameCtx);
            vsapi->freeFrame(src);
            vsapi->freeFrame(dst);
            return nullptr;
        }

        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

static void VS_CC waifu2xFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    Waifu2xData * d = static_cast<Waifu2xData *>(instanceData);
    vsapi->freeNode(d->node);
    vs_aligned_free(d->srcInterleaved);
    vs_aligned_free(d->dstInterleaved);
    vs_aligned_free(d->buffer);
    delete d->waifu2x;
    delete d;
}

static void VS_CC waifu2xCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    Waifu2xData d = {};
    int err;

    int noise = int64ToIntS(vsapi->propGetInt(in, "noise", 0, &err));
    if (err)
        noise = 1;

    d.scale = int64ToIntS(vsapi->propGetInt(in, "scale", 0, &err));
    if (err)
        d.scale = 2;

    int block = int64ToIntS(vsapi->propGetInt(in, "block", 0, &err));
    if (err)
        block = 128;

    const bool photo = !!vsapi->propGetInt(in, "photo", 0, &err);

    bool cudnn = !!vsapi->propGetInt(in, "cudnn", 0, &err);
    if (err)
        cudnn = true;

    const bool tta = !!vsapi->propGetInt(in, "tta", 0, &err);

    if (noise < 0 || noise > 2) {
        vsapi->setError(out, "Waifu2x-caffe: noise must be set to 0, 1 or 2");
        return;
    }

    if (d.scale < 1 || !isPowerOf2(d.scale)) {
        vsapi->setError(out, "Waifu2x-caffe: scale must be greater than or equal to 1 and be a power of 2");
        return;
    }

    if (block < 1) {
        vsapi->setError(out, "Waifu2x-caffe: block must be greater than or equal to 1");
        return;
    }

    d.node = vsapi->propGetNode(in, "clip", 0, nullptr);
    d.vi = *vsapi->getVideoInfo(d.node);

    if (noise == 0 && d.scale == 1) {
        vsapi->propSetNode(out, "clip", d.node, paReplace);
        vsapi->freeNode(d.node);
        return;
    }

    if (!isConstantFormat(&d.vi) || d.vi.format->sampleType != stFloat || d.vi.format->bitsPerSample != 32) {
        vsapi->setError(out, "Waifu2x-caffe: only constant format 32-bit float input supported");
        vsapi->freeNode(d.node);
        return;
    }

    VSPlugin * fmtcPlugin = vsapi->getPluginById("fmtconv", core);
    if (d.scale != 1 && d.vi.format->subSamplingW != 0 && !fmtcPlugin) {
        vsapi->setError(out, "Waifu2x-caffe: the fmtconv plugin is required for fixing horizontal chroma shift");
        vsapi->freeNode(d.node);
        return;
    }

    int iterTimesTwiceScaling;
    if (d.scale != 1) {
        d.vi.width *= d.scale;
        d.vi.height *= d.scale;
        iterTimesTwiceScaling = static_cast<int>(std::log2(d.scale));
    }

    if (d.vi.format->colorFamily == cmRGB) {
        d.srcInterleaved = vs_aligned_malloc<float>(vsapi->getVideoInfo(d.node)->width * vsapi->getVideoInfo(d.node)->height * 3 * sizeof(float), 32);
        d.dstInterleaved = vs_aligned_malloc<float>(d.vi.width * d.vi.height * 3 * sizeof(float), 32);
        if (!d.srcInterleaved || !d.dstInterleaved) {
            vsapi->setError(out, "Waifu2x-caffe: malloc failure (srcInterleaved/dstInterleaved)");
            vsapi->freeNode(d.node);
            return;
        }
    } else {
        d.buffer = vs_aligned_malloc<float>(d.vi.width * d.vi.height * sizeof(float), 32);
        if (!d.buffer) {
            vsapi->setError(out, "Waifu2x-caffe: malloc failure (buffer)");
            vsapi->freeNode(d.node);
            return;
        }
    }

    const char * mode = (d.scale == 1) ? "noise" : ((noise == 0) ? "scale" : "noise_scale");

    const std::string pluginPath(vsapi->getPluginPath(vsapi->getPluginById("com.holywu.waifu2x-caffe", core)));
    std::string modelPath(pluginPath.substr(0, pluginPath.find_last_of('/')));
    if (d.vi.format->colorFamily == cmRGB) {
        if (photo)
            modelPath += "/models/photo";
        else
            modelPath += "/models/anime_style_art_rgb";
    } else {
        modelPath += "/models/anime_style_art";
    }

    d.waifu2x = new Waifu2x();
    char * argv[] = { "" };

    const Waifu2x::eWaifu2xError waifu2xError = d.waifu2x->init(1, argv, mode, noise, d.scale, modelPath, cudnn ? "cudnn" : "gpu", boost::optional<int>(), 32, tta, block, 1);
    if (waifu2xError != Waifu2x::eWaifu2xError_OK) {
        const char * err;

        switch (waifu2xError) {
        case Waifu2x::eWaifu2xError_InvalidParameter:
            err = "invalid parameter";
            break;
        case Waifu2x::eWaifu2xError_FailedOpenModelFile:
            err = "failed open model file";
            break;
        case Waifu2x::eWaifu2xError_FailedParseModelFile:
            err = "failed parse model file";
            break;
        case Waifu2x::eWaifu2xError_FailedConstructModel:
            err = "failed construct model";
            break;
        case Waifu2x::eWaifu2xError_FailedCudaCheck:
            err = "failed cuda check";
            break;
        }

        vsapi->setError(out, (std::string("Waifu2x-caffe: ") + err + " at initialization").c_str());
        vsapi->freeNode(d.node);
        delete d.waifu2x;
        return;
    }

    Waifu2xData * data = new Waifu2xData(d);

    vsapi->createFilter(in, out, "Waifu2x-caffe", waifu2xInit, waifu2xGetFrame, waifu2xFree, fmParallelRequests, 0, data, core);

    if (d.scale != 1 && d.vi.format->subSamplingW != 0) {
        const double offset = 0.5 * (1 << d.vi.format->subSamplingW) - 0.5;
        double shift = 0.;
        for (int times = 0; times < iterTimesTwiceScaling; times++)
            shift = shift * 2. + offset;

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

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("com.holywu.waifu2x-caffe", "caffe", "Image Super-Resolution using Deep Convolutional Neural Networks", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("Waifu2x",
                 "clip:clip;"
                 "noise:int:opt;"
                 "scale:int:opt;"
                 "block:int:opt;"
                 "photo:int:opt;"
                 "cudnn:int:opt;"
                 "tta:int:opt;",
                 waifu2xCreate, nullptr, plugin);
}
