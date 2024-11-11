#hipcc  -I/opt/rocm/include -I/opt/rocm/include/roctracer -g stresstest.cc -L/opt/rocm/roctracer/lib  -v -o stresstest^
echo $PATH
#AMDCGN_INSTRUMENTATION_FUNCTIONS_FILE=/work1/amd/klowery/instrument-amdgpu-kernels/dh_comms/build/lib/hip/dh_comms_dev.gfx90a.bc hipcc -fpass-plugin=/work1/amd/klowery/instrument-amdgpu-kernels/build/hip-static/libAMDGCNMemTrace.so -I/opt/rocm/include -I/opt/rocm/include/roctracer  -O3 quicktest.cc -L/opt/rocm/roctracer/lib  -v -o quicktest
hipcc -fpass-plugin=/work1/amd/klowery/instrument-amdgpu-kernels/build/lib/libAMDGCNMemTraceHip.so --offload-arch=gfx90a -I/opt/rocm/include -I/opt/rocm/include/roctracer  -O3 quicktest.cc -L/opt/rocm/roctracer/lib -ldh_comms -v -o quicktest
#hipcc  -I/opt/rocm/include -I/opt/rocm/include/roctracer -g quicktest.cc -L/opt/rocm/roctracer/lib  -v -o quicktest
