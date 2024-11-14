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

void memory_heatmap_wrapper::report()
{
    std::cerr << "omniprobe memory heatmap report for kernel " << strKernel_ << " dispatch[" << std::dec << dispatch_id_ << "] \n";
    wrapped_.report();
}

void memory_heatmap_wrapper::clear()
{
    wrapped_.clear();
}

