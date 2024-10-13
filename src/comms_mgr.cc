#include "inc/comms_mgr.h"
#include "inc/hsa_mem_mgr.h"

comms_mgr::comms_mgr(HsaApiTable *pTable) : kern_arg_allocator_(pTable, std::cerr), pTable_(pTable)
{
}
comms_mgr::~comms_mgr()
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto item : mem_mgrs_)
    {
        delete item.second;
    }
    mem_mgrs_.clear();
}

dh_comms::dh_comms * comms_mgr::checkoutCommsObject(hsa_agent_t agent)
{
    std::lock_guard<std::mutex> lock(mutex_);
    return NULL;
}

bool comms_mgr::checkinCommsObject(hsa_agent_t agent, dh_comms::dh_comms *object)
{
    std::lock_guard<std::mutex> lock(mutex_);
    return false;
}

    
bool comms_mgr::addAgent(hsa_agent_t agent)
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<pool_specs_t> pools;
    struct parms {
        std::vector<pool_specs_t> *pools;
        hsa_agent_t agent;
    };
    struct parms data = {&pools, agent};
    hsa_amd_agent_iterate_memory_pools(agent, [](hsa_amd_memory_pool_t pool, void *data){
        hsa_status_t status;
        struct parms *parms = reinterpret_cast<struct parms *>(data);
        size_t granularity = 0;
        hsa_amd_segment_t segment;
        uint32_t flags;
        bool runtime_allocatable;
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
        if (HSA_AMD_SEGMENT_GLOBAL != segment)
            return HSA_STATUS_SUCCESS;
        status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &runtime_allocatable);
        if (status == HSA_STATUS_SUCCESS && runtime_allocatable)
        {
            status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,&granularity);
            if (status != HSA_STATUS_SUCCESS)
                throw std::bad_alloc();
            size_t size;
            status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SIZE, &size);
            if (status != HSA_STATUS_SUCCESS)
                throw std::bad_alloc();
        }
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
        if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED)
        {
            parms->pools->push_back({parms->agent, pool, granularity});
        }

        return HSA_STATUS_SUCCESS;
    }, &data);

    if (pools.size())
    {
        for (auto item : pools)
        {
            hsa_mem_mgr * mgr = new hsa_mem_mgr(item.agent_, item, kern_arg_allocator_);
            mem_mgrs_[item.agent_] = mgr;
        }
    }
    return false;
}


bool comms_mgr::growBufferPool(hsa_agent_t agent, size_t count)
{
    return false;
}


default_message_processor::default_message_processor(comms_mgr *mgr)
{
}

default_message_processor::~default_message_processor()
{
}

size_t default_message_processor::operator()(char *&message_p, size_t size, size_t sub_buf_no)
{
    cerr << "default_message_processor:\n\tMessage of size " << std::dec << size << " with " << sub_buf_no << " sub buffers\n";
    return size;
}


bool default_message_processor::is_thread_safe() const
{
    return false;
}
