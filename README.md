Description
===========

waifu2x filter for VapourSynth, based on [waifu2x-caffe](https://github.com/lltcggie/waifu2x-caffe).


Note
====

The folder `models` must be located in the same folder as `Waifu2x-caffe.dll`.

Requires `cudnn64_7.dll` (cuDNN v7.3.1) to be in the search path. Due to the license of cuDNN, I can't distribute the required dll file. You have to register and download yourself at https://developer.nvidia.com/cudnn.


Usage
=====

    caffe.Waifu2x(clip clip[, int noise=0, int scale=2, int block_w=128, int block_h=block_w, int model=3, bint cudnn=True, int processor=0, bint tta=False, int batch=1])

* clip: Clip to process. Only planar format with float sample type of 32 bit depth is supported.

* noise: Noise reduction level.
  * -1 = none
  * 0 = low
  * 1 = medium
  * 2 = high
  * 3 = highest

* scale: Upscaling factor. Must be a power of 2. Set to 1 for no upscaling.

* block_w: The horizontal block size for dividing the image during processing. Smaller value results in lower VRAM usage, while larger value may not necessarily give faster speed. The optimal value may vary according to different graphics card and image size.

* block_h: The same as `block_w` but for vertical.

* model: Specifies which model to use. Only `anime_style_art` is Y model (luma only), the others are RGB models.
  * 0 = anime_style_art (for 2D illustration)
  * 1 = anime_style_art_rgb (for 2D illustration)
  * 2 = photo (for photo and anime)
  * 3 = upconv_7_anime_style_art_rgb (has faster speed than `anime_style_art_rgb`, with equal or better quality)
  * 4 = upconv_7_photo (has faster speed than `photo`, with equal or better quality)
  * 5 = upresnet10 (for 2D illustration). Note that the result will change when the block size is changed. The recommended block size is 38 for this model. Use a larger `batch` size to compensate the slowness due to small block size.

* cudnn: When set to true, it uses cuDNN for processing. When set to false, CUDA will be used instead.

* processor: Specifies which GPU device to use. The device number begins with 0. The default device will be used if a nonexistent device is specified.

* tta: Whether TTA(Test-Time Augmentation) mode is used. It increases PSNR by 0.15 or so, but 8 times slower.

* batch: The batch size for simultaneous processing of blocks.


Compilation
===========

Requires [customized Caffe library](https://github.com/HolyWu/caffe). To build `Caffe`, you must have all the dependencies: `CUDA Toolkit 10`, `cuDNN 7`, `OpenBLAS`, `Boost`, `protobuf`, `glog`, and `gflags`. Additionally, `waifu2x-caffe` library requires `OpenCV 3`. The defaults in `Makefile.config` should work, but modify the relevant lines if it doesn't (`CUDA_DIR` especially). Then just type `make all -j4` to compile `Caffe`.

```
./autogen.sh
./configure
make
```
