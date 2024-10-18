g++ -g -I /opt/rocm-6.2.1/include/amd_comgr/ -I /opt/rocm-6.2.1/include/hsa hip_test.cc -o hip_test -L /opt/rocm-6.2.1/lib -lamd_comgr -lhsa-runtime64 
