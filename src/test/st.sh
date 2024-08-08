hipcc -I/opt/rocm/include -I/opt/rocm/include/roctracer -g stresstest.cc -L/opt/rocm/roctracer/lib  -v -o stresstest
hipcc -I/opt/rocm/include -I/opt/rocm/include/roctracer -g quicktest.cc -L/opt/rocm/roctracer/lib  -v -o quicktest
