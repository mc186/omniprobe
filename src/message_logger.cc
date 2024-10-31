#include <cstdio>
#include <iostream>
#include <vector>
#include <cassert>
#include "inc/message_logger.h"
#include "data_headers.h"
#include "message.h"

message_logger_t::message_logger_t(const std::string& strKernel, uint64_t dispatch_id, bool verbose)
    : strKernel_(strKernel),
      dispatch_id_(dispatch_id),
      verbose_(verbose)
{
}

message_logger_t::~message_logger_t()
{
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
    for (size_t i = 0; i != message.no_data_items(); ++i)
    {
        uint64_t address = *(const uint64_t *)message.data_item(i);
        if (verbose_)
        {
            //printf("memory_heatmap: added address 0x%lx to map\n", address);
        }
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

