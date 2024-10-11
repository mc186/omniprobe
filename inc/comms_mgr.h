
#pragma once

#include <shared_mutex>
#include <memory>
#include <dlfcn.h>
#include <cxxabi.h>

#include <hsa.h>
#include <hsa_ven_amd_aqlprofile.h>
#include <hsa_ven_amd_loader.h>
#include "rocprofiler/rocprofiler.h"
#define AMD_INTERNAL_BUILD
#include <hsa_api_trace.h>
#include <set>
#include <tuple>
#include <atomic>

#include "timehelper.h"
#include "utils.h"
#include "dh_comms.h"
#include "hsa_mem_mgr.h"
class comms_mgr 
{
public: 
    comms_mgr(HsaApiTable *pTable);
    ~comms_mgr();
    void * checkoutCommsDescriptor(hsa_agent_t agent);
    bool checkinCommsDescriptor(hsa_agent_t agent, void *device_buffer);
    bool addAgent(hsa_agent_t agent);
private:
    std::mutex mutex_;
    bool growBufferPool(hsa_agent_t agent, size_t count);
    std::map<void *, dh_comms::dh_comms_descriptor> pending_buffers_;
    std::map<hsa_agent_t, hsa_amd_memory_pool_t, hsa_cmp<hsa_agent_t>> mem_pools_;
    std::map<hsa_agent_t, hsa_mem_mgr *, hsa_cmp<hsa_agent_t>> mem_mgrs_;
    std::map<hsa_agent_t, std::vector<void *>, hsa_cmp<hsa_agent_t>> device_buffer_pool_;
    std::map<hsa_agent_t, std::vector<dh_comms::dh_comms_descriptor>, hsa_cmp<hsa_agent_t>> descriptor_pool_;
    HsaApiTable *pTable_;
};

