#pragma once

#include <map>
#include "message_handlers.h"

class message_logger_t : public dh_comms::message_handler_base
{
public:
    message_logger_t(const std::string& strKernel, uint64_t dispatch_id, bool verbose = false);
    message_logger_t(const message_logger_t&) = default;
    virtual ~message_logger_t();
    virtual bool handle(const dh_comms::message_t &message) override;
    virtual void report() override;
    virtual void clear() override;

private:
    std::string strKernel_;
    uint64_t dispatch_id_;
    bool verbose_;
    // out iostream here
};

