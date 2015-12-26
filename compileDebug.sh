#g++ -std=c++11 -O3 -o main sun3d_reader.cpp `pkg-config --cflags --libs opencv` -I. -I/usr/include/ -I/usr/include/pcl-1.7/ -I/usr/include/eigen3/ -I/usr/include/vtk-5.8/ -L/usr/lib -L/usr/lib/x86_64-linux-gnu/ -L/usr/local/cuda-7.0/lib64 -lcuda -lcudart -lpcl_apps -lpcl_common -lpcl_features -lpcl_filters -lpcl_io -lpcl_kdtree -lpcl_keypoints -lpcl_octree -lpcl_registration -lpcl_sample_consensus -lpcl_search -lpcl_segmentation -lpcl_surface -lpcl_visualization -lpng -ljpeg -lcurl -DBOOST_SYSTEM_NO_DEPRECATED -Wno-deprecated -Wno-write-strings -Wl,-rpath=/usr/local/cuda-7.0/lib64/ -Wl,-rpath=/usr/include
#!/bin/bash
# export PATH=$PATH:/usr/local/cuda/bin
g++ -std=c++11 -O3 -o main main.cpp -I. -I/usr/include/ -I./pc2tsdf/ -L/usr/lib -L/usr/lib/x86_64-linux-gnu/
g++ -std=c++11 -g -Og -o matcher matcher.cpp -I. -I/usr/include/ -I./pc2tsdf/ -L/usr/lib -L/usr/lib/x86_64-linux-gnu/