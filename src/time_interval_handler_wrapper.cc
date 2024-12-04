#include "inc/time_interval_handler_wrapper.h"
#include <iostream>
#include <cassert>
#include "time_interval_handler.h"

time_interval_handler_wrapper::time_interval_handler_wrapper(const std::string& strKernel, uint64_t dispatch_id, bool verbose)
    : first_start_(0xffffffffffffffff),
      last_stop_(0),
      total_time_(0),
      no_intervals_(0),
      verbose_(verbose),
      strKernel_(strKernel),
      dispatch_id_(dispatch_id),
      wrapped_(verbose)
{
}

time_interval_handler_wrapper::~time_interval_handler_wrapper()
{
}

bool time_interval_handler_wrapper::handle(const dh_comms::message_t &message, const std::string& kernel, kernelDB::kernelDB& kdb)
{
    auto instructions = kdb.getInstructionsForLine(kernel,message.wave_header().src_loc_idx);
    for (auto inst : instructions)
        std::cout << inst.inst_ << std::endl;
    return handle(message);
}

bool time_interval_handler_wrapper::handle(const dh_comms::message_t &message)
{
    return wrapped_.handle(message);
}

void time_interval_handler_wrapper::report()
{
    std::cerr << "omniprobe time_interval report for kernel " << strKernel_ << " dispatch[" << std::dec << dispatch_id_ << "]\n";
    wrapped_.report();
}

void time_interval_handler_wrapper::clear()
{
    wrapped_.clear();
    strKernel_ = "";
    dispatch_id_ = 0;
}

