#include "inc/memory_analysis_wrapper.h"

#include "hip_utils.h"
#include "utils.h"

#include <cassert>
#include <hip/hip_runtime.h>
#include <set>
#include <string>
#include <iostream>

memory_analysis_wrapper_t::memory_analysis_wrapper_t(const std::string& kernel, uint64_t dispatch_id, const std::string& location,  bool verbose) : 
    kernel_(kernel), dispatch_id_(dispatch_id), location_(location), wrapped_(verbose)
{
}

bool memory_analysis_wrapper_t::handle(const dh_comms::message_t &message, const std::string& kernel, kernelDB::kernelDB& kdb)
{
    auto instructions = kdb.getInstructionsForLine(kernel,message.wave_header().src_loc_idx);
    for (auto inst : instructions)
        std::cout << inst.inst_ << std::endl;
    return handle(message);
}


bool memory_analysis_wrapper_t::handle(const dh_comms::message_t &message) {
  return wrapped_.handle(message);
}

void memory_analysis_wrapper_t::report(const std::string& kernel_name, kernelDB::kernelDB& kdb)
{
    if (kernel_name.length() == 0)
    {
        std::vector<uint32_t> lines;
        kdb.getKernelLines(kernel_name, lines);
    }
    report();
}

void memory_analysis_wrapper_t::report() {
  std::cout << "Memory analysis for " << kernel_ << " dispatch_id[" << std::dec << dispatch_id_ << "]" << std::endl;
  wrapped_.report();
}

void memory_analysis_wrapper_t::clear() {
  wrapped_.clear();
}
