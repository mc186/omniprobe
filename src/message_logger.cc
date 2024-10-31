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
}

message_logger_t::~message_logger_t()
{
    if(location_ != "console")
        delete log_file_;
}

bool message_logger_t::handle(const dh_comms::message_t &message)
{
    if ((dh_comms::message_type)message.wave_header().user_type != dh_comms::message_type::address)
    {
        if (verbose_)
        {
            printf("message_logger: skipping message with user type 0x%x\n", message.wave_header().user_type);
        }
        return false;
    }
    assert(message.data_item_size() == sizeof(uint64_t));

    dh_comms::wave_header_t hdr = message.wave_header();


    *log_file_ << "\"" << strKernel_ << "\"," << std::dec << dispatch_id_ << ",";

    *log_file_ << "0x" << std::hex << std::setw(sizeof(void*) * 2) << std::setfill('0') << message.wave_header().exec << ",";
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

void message_logger_t::report()
{
    printf("Message Logger complete.\n");
}

void message_logger_t::clear()
{
}

