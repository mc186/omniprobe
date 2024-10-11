#include "inc/comms_mgr.h"

comms_mgr::comms_mgr(HsaApiTable *pTable) : pTable_(pTable)
{
}
comms_mgr::~comms_mgr()
{
}

void * comms_mgr::checkoutCommsDescriptor(hsa_agent_t agent)
{
    std::lock_guard<std::mutex> lock(mutex_);
    return NULL;
}

bool comms_mgr::checkinCommsDescriptor(hsa_agent_t agent, void *device_buffer)
{
    std::lock_guard<std::mutex> lock(mutex_);
    return false;
}

    
bool comms_mgr::addAgent(hsa_agent_t agent)
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::cerr << "Add Agent called for comms_mgr\n";
    hsa_amd_agent_iterate_memory_pools(agent, [](hsa_amd_memory_pool_t pool, void *data){
        hsa_status_t status;
        hsa_amd_segment_t segment;
        uint32_t flags;
        bool runtime_allocatable;
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
        if (HSA_AMD_SEGMENT_GLOBAL != segment)
            return HSA_STATUS_SUCCESS;
        status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &runtime_allocatable);
        if (status == HSA_STATUS_SUCCESS && runtime_allocatable)
        {
            size_t granularity = 0;
            status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,&granularity);
            if (status == HSA_STATUS_SUCCESS)
            {
                std::cerr << "Runtime memory pool allocation granularity " << granularity << std::endl;
            }
            size_t size;
            status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SIZE, &size);
            if (status == HSA_STATUS_SUCCESS)
                std::cerr << "Runtime memory pool size: " << size << std::endl;
        }
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
        if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED)
        {
            std::cerr << "I found the coarse grained pot for this memory pool\n";
        }

        return HSA_STATUS_SUCCESS;
    }, this);
    return false;
}


bool comms_mgr::growBufferPool(hsa_agent_t agent, size_t count)
{
    return false;
}
