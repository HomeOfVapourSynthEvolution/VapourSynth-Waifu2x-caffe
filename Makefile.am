INCLUDE_DIRS = -I/usr/local/include -I/usr/include -I/usr/local/cuda/include -I/opt/cuda/include -I/usr/local/include/openblas -I/usr/include/openblas -I/opt/OpenBLAS/include -I$(srcdir)/Waifu2x-caffe/include -I$(srcdir)/Waifu2x-caffe/include/msgpack-c
common_cflags = -O3 -ffast-math -pthread $(INCLUDE_DIRS) $(MFLAGS)
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
LIBS = -lcaffe -lcudart -lcublas -lcurand -lcudnn -lopenblas -lboost_filesystem -lboost_iostreams -lboost_thread -lboost_system -lprotobuf -lglog -lgflags -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -lz -lgfortran -lm
libwaifu2x_caffe_la_LDFLAGS = -no-undefined -avoid-version -pthread $(LIBRARY_DIRS) $(LIBS) $(PLUGINLDFLAGS)
