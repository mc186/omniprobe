#include "inc/time_interval_handler.h"
#include <iostream>
#include <cassert>


time_interval_handler_t::time_interval_handler_t(const std::string& strKernel, uint64_t dispatch_id, bool verbose)
    : first_start_(0xffffffffffffffff),
      last_stop_(0),
      total_time_(0),
      no_intervals_(0),
      verbose_(verbose),
      strKernel_(strKernel),
      dispatch_id_(dispatch_id)
{
}

time_interval_handler_t::~time_interval_handler_t()
{
}

bool time_interval_handler_t::handle(const dh_comms::message_t &message)
{
    if ((dh_comms::message_type)message.wave_header().user_type != dh_comms::message_type::time_interval)
    {
        if (verbose_)
        {
            printf("time_interval_handler: skipping message with user type 0x%x\n", message.wave_header().user_type);
        }
        return false;
    }
    assert(message.data_item_size() == sizeof(time_interval));
    for (size_t i = 0; i != message.no_data_items(); ++i)
    {
        time_interval ti = *(const time_interval *)message.data_item(i);
        assert(ti.start <= ti.stop);

        first_start_ = ti.start < first_start_ ? ti.start : first_start_;
        last_stop_ = ti.stop > last_stop_ ? ti.stop : last_stop_;
        total_time_ += ti.stop - ti.start;
        ++no_intervals_;
        if (verbose_)
        {
            printf("time_interval processed:\n");
            printf("\tstart = %lu\n", ti.start);
            printf("\tstop = %lu\n", ti.stop);
        }
    }
    return true;
}

void time_interval_handler_t::report()
{
    if (no_intervals_ != 0)
    {
        double average_time = (double)total_time_ / no_intervals_;
        std::cerr << "omniprobe time_interval report for kernel " << strKernel_ << " dispatch[" << std::dec << dispatch_id_ << "]\n";
//        printf("time_interval report\n");
        printf("\ttotal time for all %zu intervals: %lu\n", no_intervals_, total_time_);
        printf("\taverage time per interval: %.0f\n", average_time);
        printf("\tfirst start: %lu\n", first_start_);
        printf("\tlast stop: %lu\n", last_stop_);
        printf("\ttime from first start to last stop: %lu"
               "\t   (%f times the average interval time)\n",
               last_stop_ - first_start_, (last_stop_ - first_start_) / average_time);
    }
}

void time_interval_handler_t::clear()
{
    first_start_ = 0;
    last_stop_ = 0;
    total_time_ = 0;
    no_intervals_ = 0;
    strKernel_ = "";
    dispatch_id_ = 0;
}

