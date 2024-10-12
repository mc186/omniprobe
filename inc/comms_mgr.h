
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


typedef struct pool_specs
{
    hsa_agent_t agent_;
    hsa_amd_memory_pool_t pool_;
    size_t min_alloc_size_;
}pool_specs_t;

class comms_mgr 
{
public: 
    comms_mgr(HsaApiTable *pTable);
    ~comms_mgr();
    dh_comms::dh_comms * checkoutCommsObject(hsa_agent_t agent);
    bool checkinCommsObject(hsa_agent_t agent, dh_comms::dh_comms *object);
    bool addAgent(hsa_agent_t agent);
private:
    KernArgAllocator kern_arg_allocator_;
    std::mutex mutex_;
    bool growBufferPool(hsa_agent_t agent, size_t count);
    std::map<hsa_agent_t, pool_specs_t, hsa_cmp<hsa_agent_t>> mem_pools_;
    std::map<hsa_agent_t, dh_comms::dh_comms_mem_mgr *, hsa_cmp<hsa_agent_t>> mem_mgrs_;
    std::map<hsa_agent_t, std::vector<dh_comms::dh_comms *>, hsa_cmp<hsa_agent_t>> comms_pool_;
    std::map<hsa_agent_t, std::vector<dh_comms::dh_comms *>, hsa_cmp<hsa_agent_t>> pending_comms_;
    HsaApiTable *pTable_;
};

