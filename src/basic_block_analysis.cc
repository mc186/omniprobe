#include <iostream>
#include <cassert>
#include "inc/basic_block_analysis.h"

uint32_t countSetBits(uint64_t mask) {
    uint32_t count = 0;
    while (mask) {
        count += mask & 1;
        mask >>= 1;
    }
    return count;
}

basic_block_analysis::basic_block_analysis(const std::string& strKernel, uint64_t dispatch_id, std::string& strLocation, bool verbose)
    : first_start_(0xffffffffffffffff),
      last_stop_(0),
      total_time_(0),
      no_intervals_(0),
      verbose_(verbose),
      strKernel_(strKernel),
      dispatch_id_(dispatch_id),
      current_block_(nullptr),
      start_time_(0)
{
}

basic_block_analysis::~basic_block_analysis()
{
}

bool basic_block_analysis::handle(const dh_comms::message_t &message, const std::string& kernel, kernelDB::kernelDB& kdb)
{
    auto hdr = message.wave_header();
    try
    {
        auto& instructions = kdb.getInstructionsForLine(kernel, hdr.dwarf_line);
        if (instructions.size())
        {
            if (current_block_)
            {
                auto it = block_timings_.find(current_block_);
                if (it != block_timings_.end())
                    it->second += hdr.timestamp - start_time_;
                else
                    block_timings_[current_block_] = hdr.timestamp - start_time_;
            }
            current_block_ = instructions[0].block_;
            start_time_ = hdr.timestamp;
        }
        for (auto inst : instructions)
            std::cout << inst.inst_ << std::endl;
    }
    catch (std::runtime_error e)
    {
    }
    return handle(message);
}

bool basic_block_analysis::handle(const dh_comms::message_t &message)
{
    return true;
}

void basic_block_analysis::report(const std::string& kernel_name, kernelDB::kernelDB& kdb)
{
    if (kernel_name.length() == 0)
    {
        std::vector<uint32_t> lines;
        kdb.getKernelLines(kernel_name, lines);
    }
    report();
}

void basic_block_analysis::report()
{
    std::cerr << "omniprobe time_interval report for kernel " << strKernel_ << " dispatch[" << std::dec << dispatch_id_ << "]\n";

}

void basic_block_analysis::clear()
{
    strKernel_ = "";
    dispatch_id_ = 0;
}
