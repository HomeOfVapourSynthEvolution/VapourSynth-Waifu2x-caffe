project('Waifu2x-caffe', 'cpp',
  default_options : ['buildtype=release', 'b_ndebug=if-release', 'cpp_std=c++14'],
  meson_version : '>=0.48.0',
  version : '13'
)

add_project_arguments('-ffast-math', language : 'cpp')

sources = [
  'Waifu2x-caffe/cNet.cpp',
  'Waifu2x-caffe/cNet.h',
  'Waifu2x-caffe/stImage.cpp',
  'Waifu2x-caffe/stImage.h',
  'Waifu2x-caffe/waifu2x.cpp',
  'Waifu2x-caffe/waifu2x.h',
  'Waifu2x-caffe/Waifu2x-caffe.cpp'
]

cc = meson.get_compiler('cpp')

vapoursynth_dep = dependency('vapoursynth').partial_dependency(compile_args : true, includes : true)

deps = [vapoursynth_dep]

deps += cc.find_library('caffe', dirs : get_option('caffe_libdir'))

cuda_libdir = get_option('cudalibdir')
deps += cc.find_library('cudart', dirs : cuda_libdir)
deps += cc.find_library('cublas', dirs : cuda_libdir)
deps += cc.find_library('curand', dirs : cuda_libdir)
deps += cc.find_library('cudnn', dirs : cuda_libdir)

deps += dependency('openblas')

deps += dependency('boost', modules : ['filesystem', 'iostreams', 'thread', 'system'])

deps += dependency('protobuf')

deps += dependency('libglog')

deps += dependency('gflags')

deps += dependency('opencv4')

deps += dependency('zlib')

deps += dependency('threads')

includes = [
  include_directories(get_option('caffe_includedir')),
  include_directories(get_option('cudaincludedir')),
  include_directories('Waifu2x-caffe/include'),
  include_directories('Waifu2x-caffe/include/msgpack-c')
]

if host_machine.cpu_family().startswith('x86')
  add_project_arguments('-mfpmath=sse', '-msse2', language : 'cpp')
endif

vapoursynth_plugindir = join_paths(vapoursynth_dep.get_pkgconfig_variable('libdir'), 'vapoursynth')

shared_module('waifu2x-caffe', sources,
  dependencies : deps,
  include_directories : includes,
  install : true,
  install_dir : vapoursynth_plugindir,
  gnu_symbol_visibility : 'hidden'
)

install_subdir('Waifu2x-caffe/models',
  install_dir : vapoursynth_plugindir
)
