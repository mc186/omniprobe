
/******************************************************************************
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*******************************************************************************/
#include <cstdio>
#include <iostream>
#include <vector>
#include <cassert>
#include "inc/memory_heatmap_wrapper.h"
#include "data_headers.h"
#include "message.h"
#include "memory_heatmap.h"

memory_heatmap_wrapper::memory_heatmap_wrapper(const std::string& strKernel, uint64_t dispatch_id, size_t page_size, bool verbose)
    : strKernel_(strKernel),
      dispatch_id_(dispatch_id),
      verbose_(verbose),
      page_size_(page_size),
      wrapped_(page_size, verbose)
{
}

memory_heatmap_wrapper::~memory_heatmap_wrapper()
{
}

bool memory_heatmap_wrapper::handle(const dh_comms::message_t &message, const std::string& kernel, kernelDB::kernelDB& kdb)
{
    return handle(message);
}

bool memory_heatmap_wrapper::handle(const dh_comms::message_t &message)
{
    if (message.wave_header().user_type != dh_comms::message_type::address)
    {
        if (verbose_)
        {
            printf("memory_heatmap: skipping message with user type 0x%x\n", message.wave_header().user_type);
        }
        return false;
    }
    return wrapped_.handle(message);;
}
void memory_heatmap_wrapper::report(const std::string& kernel_name, kernelDB::kernelDB& kdb)
{
    if (kernel_name.length() == 0)
    {
        std::vector<uint32_t> lines;
        kdb.getKernelLines(kernel_name, lines);
    }
    report();
}

void memory_heatmap_wrapper::report()
{
    std::cerr << "omniprobe memory heatmap report for kernel " << strKernel_ << " dispatch[" << std::dec << dispatch_id_ << "] \n";
    wrapped_.report();
}

void memory_heatmap_wrapper::clear()
{
    wrapped_.clear();
}
