Description
===========

waifu2x filter for VapourSynth, based on [waifu2x-caffe](https://github.com/lltcggie/waifu2x-caffe).


Note
====

The folder `models` must be located in the same folder as `Waifu2x-caffe.dll`.

Requires `cudnn64_5.dll` (cuDNN v5.1 RC) to be in the search path. Due to the license of cuDNN, I can't distribute the required dll file. You have to register and download yourself at https://developer.nvidia.com/cudnn.


Usage
=====

    caffe.Waifu2x(clip clip[, int noise=1, int scale=2, int block_w=128, int block_h=block_w, int model=1, bint cudnn=True, int processor=0, bint tta=False])

* clip: The input clip. Must be 32-bit floating point. Any planar color family is supported.

* noise: Noise reduction level.
  * 0 = none
  * 1 = medium
  * 2 = high
  * 3 = highest

* scale: Upscaling factor. Must be a power of 2. Set to 1 for no upscaling.

* block_w: The horizontal block size for dividing the image during processing. Smaller value results in lower VRAM usage, while larger value may not necessarily give faster speed. The optimal value may vary according to different graphics card and image size.

* block_h: The same as `block_w` but for vertical.

* model: Specifies which model to use.
  * 0 = anime_style_art
  * 1 = anime_style_art_rgb
  * 2 = photo
  * 3 = upconv_7_anime_style_art_rgb
  * 4 = upconv_7_photo

* cudnn: When set to true, it uses cuDNN for processing. When set to false, CUDA will be used instead.

* processor: Specifies which GPU device to use. The device number begins with 0. The default device will be used if a nonexistent device is specified.

* tta: Whether TTA(Test-Time Augmentation) mode is used. It increases PSNR by 0.15 or so, but 8 times slower.
