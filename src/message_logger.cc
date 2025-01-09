#include <cstdio>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cassert>
#include "inc/message_logger.h"
#include "data_headers.h"
#include "message.h"

message_logger_t::message_logger_t(const std::string& strKernel, uint64_t dispatch_id, std::string& location, bool verbose)
    : strKernel_(strKernel),
      dispatch_id_(dispatch_id),
      location_(location),
      verbose_(verbose)
{
    location_ = location;
    if (location == "console")
        log_file_ = &std::cout;
    else
        log_file_ = new std::ofstream(location, std::ios::app);
    
    *log_file_ << "ADDRESS_MESSAGE,timestamp,kernel,src_line,dispatch,exec_mask,xcc_id,se_id,cu_id,kind,address" << std::endl;
}

message_logger_t::~message_logger_t()
{
    if(location_ != "console")
        delete log_file_;
}
bool message_logger_t::handle(const dh_comms::message_t &message, const std::string& kernel, kernelDB::kernelDB& kdb)
{
    std::cout << kernel << std::endl;
    dh_comms::wave_header_t hdr = message.wave_header();
    if (message.wave_header().user_type != dh_comms::message_type::address)
        return false;
    try
    {
    auto instructions = kdb.getInstructionsForLine(kernel,hdr.src_loc_idx);
    for (auto inst : instructions)
        std::cout << inst.inst_ << std::endl;
    }
    catch (std::runtime_error e)
    {
    }
    return handle(message);
}

bool message_logger_t::handle(const dh_comms::message_t &message)
{
    if (message.wave_header().user_type != dh_comms::message_type::address)
    {
        if (verbose_)
        {
            printf("message_logger: skipping message with user type 0x%x\n", message.wave_header().user_type);
        }
        return false;
    }
    assert(message.data_item_size() == sizeof(uint64_t));

    dh_comms::wave_header_t hdr = message.wave_header();



    *log_file_ << "ADDRESS_MESSAGE," << std::dec << hdr.timestamp << ",\"" << strKernel_ << "\"," << hdr.src_loc_idx << "," << dispatch_id_ << ",";


    *log_file_ << "0x" << std::hex << std::setw(sizeof(void*) * 2) << std::setfill('0') << hdr.exec << "," << std::dec << hdr.xcc_id << "," << hdr.se_id << "," << hdr.cu_id << ",";
    *log_file_ << (hdr.user_data & 0b11) << ",";
    for (size_t i = 0; i != message.no_data_items(); ++i)
    {
        uint64_t address = *(const uint64_t *)message.data_item(i);
        if (verbose_)
        {
            //printf("memory_heatmap: added address 0x%lx to map\n", address);
        }

        *log_file_ << "0x" << std::hex << std::setw(sizeof(void*) * 2) << std::setfill('0') << address;
        if (i < message.no_data_items() - 1)
            *log_file_ << ",";
        else
            *log_file_ << std::endl;
    }
    return true;
}

void message_logger_t::report(const std::string& kernel_name, kernelDB::kernelDB& kdb)
{
    if (kernel_name.length() == 0)
    {
        std::vector<uint32_t> lines;
        kdb.getKernelLines(kernel_name, lines);
    }
    report();
}

void message_logger_t::report()
{
    printf("Omniprobe Message Logger complete.\n");
}

void message_logger_t::clear()
{
}

