CXX=g++
CXX_FLAGS=-std=c++11 -m64 -fabi-version=6 -march=native -O3
CXX_INCLUDE=-lVc

all: scalar

scalar:
	$(CXX) vc.cc -o bin/vc_scalar $(CXX_FLAGS) $(CXX_INCLUDE)

avx:
	$(CXX) vc.cc -o bin/vc_avx $(CXX_FLAGS) $(CXX_INCLUDE) -mavx

sse:
	$(CXX) vc.cc -o bin/vc_sse $(CXX_FLAGS) $(CXX_INCLUDE) -msse