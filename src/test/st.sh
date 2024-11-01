#hipcc  -I/opt/rocm/include -I/opt/rocm/include/roctracer -g stresstest.cc -L/opt/rocm/roctracer/lib  -v -o stresstest
#hipcc -v -fpass-plugin=/work1/amd/klowery/instrument-amdgpu-kernels/build/lib/libAMDGCNMemTrace.so -I/opt/rocm/include -I/opt/rocm/include/roctracer -g quicktest.cc -L/opt/rocm/roctracer/lib  -v -o quicktest
hipcc  -I/opt/rocm/include -I/opt/rocm/include/roctracer -g quicktest.cc -L/opt/rocm/roctracer/lib  -v -o quicktest
