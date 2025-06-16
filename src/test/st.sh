#hipcc  -I/opt/rocm/include -I/opt/rocm/include/roctracer -g stresstest.cc -L/opt/rocm/roctracer/lib  -v -o stresstest^
echo $PATH
#AMDCGN_INSTRUMENTATION_FUNCTIONS_FILE=/work1/amd/klowery/instrument-amdgpu-kernels/dh_comms/build/lib/hip/dh_comms_dev.gfx90a.bc hipcc -fpass-plugin=/work1/amd/klowery/instrument-amdgpu-kernels/build/hip-static/libAMDGCNMemTrace.so -I/opt/rocm/include -I/opt/rocm/include/roctracer  -O3 quicktest.cc -L/opt/rocm/roctracer/lib  -v -o quicktest
#hipcc -L/home1/klowery/.local/lib -ldh_comms -Rpass-analysis=kernel-resource-usage -fpass-plugin=/work1/amd/klowery/logduration/omniprobe/lib/libAMDGCNMemTraceHIP.so -I/home1/klowery/.local/include/dh_comms -fgpu-rdc --offload-arch=gfx90a  -std=c++17 -ggdb -O3  quicktest.cc  -v  -o quicktest 
hipcc -fpass-plugin=/work/logduration/build/external/instrument-amdgpu-kernels-rocm/build/lib/libAMDGCNSubmitBBStart-rocm.so -fgpu-rdc --offload-arch=gfx1030  -std=c++17 -ggdb  quicktest.cc -o quicktest
#hipcc  -I/opt/rocm/include -I/opt/rocm/include/roctracer -g quicktest.cc -L/opt/rocm/roctracer/lib  -v -o quicktest
