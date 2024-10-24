#pragma once

#include <map>
#include "message_handlers.h"

class memory_heatmap_t : public dh_comms::message_handler_base
{
public:
    memory_heatmap_t(std::string& strKernel, uint64_t dispatch_id, size_t page_size = 1024 * 1024, bool verbose = false);
    memory_heatmap_t(const memory_heatmap_t&) = default;
    virtual ~memory_heatmap_t();
    virtual bool handle(const dh_comms::message_t &message) override;
    virtual void report() override;
    virtual void clear() override;

private:
    std::string strKernel_;
    uint64_t dispatch_id_;
    bool verbose_;
    size_t page_size_;
    std::map<uint64_t, size_t> page_counts_;
};

