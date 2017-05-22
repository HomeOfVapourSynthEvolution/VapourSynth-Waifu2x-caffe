INCLUDE_DIRS = -I/usr/local/include -I/usr/include -I/usr/local/cuda/include -I/opt/cuda/include -I/usr/local/include/openblas -I/usr/include/openblas -I/opt/OpenBLAS/include -I$(srcdir)/Waifu2x-caffe/include -I$(srcdir)/Waifu2x-caffe/include/msgpack-c
common_cflags = -O3 -fopenmp -ffast-math -DUSE_CUDNN $(INCLUDE_DIRS) $(MFLAGS)
AM_CXXFLAGS = -std=c++11 $(common_cflags)

AM_CPPFLAGS = $(VapourSynth_CFLAGS)

lib_LTLIBRARIES = libwaifu2x-caffe.la

libwaifu2x_caffe_la_SOURCES = Waifu2x-caffe/cNet.cpp \
                              Waifu2x-caffe/cNet.h \
                              Waifu2x-caffe/stImage.cpp \
                              Waifu2x-caffe/stImage.h \
                              Waifu2x-caffe/waifu2x.cpp \
                              Waifu2x-caffe/waifu2x.h \
                              Waifu2x-caffe/Waifu2x-caffe.cpp

LIBRARY_DIRS = -L/usr/local/lib -L/usr/lib -L/usr/local/cuda/lib64 -L/opt/cuda/lib64 -L/opt/OpenBLAS/lib
LIBS = -lcudart -lcublas -lcurand -lglog -lgflags -lprotobuf -lboost_system -lboost_filesystem -lboost_thread -lboost_iostreams -lm -lhdf5_hl -lhdf5 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lcudnn -lopenblas_openmp -lz -lcaffe
libwaifu2x_caffe_la_LDFLAGS = -no-undefined -avoid-version -fopenmp $(LIBRARY_DIRS) $(LIBS) $(PLUGINLDFLAGS)
