#include "inc/hsa_mem_mgr.h"


hsa_mem_mgr::hsa_mem_mgr(hsa_agent_t agent, const pool_specs_t& pool, const KernArgAllocator& allocator) : dh_comms::dh_comms_mem_mgr(), agent_(agent), pool_(pool), allocator_(allocator)
{
}

hsa_mem_mgr::~hsa_mem_mgr()
{
}

void * hsa_mem_mgr::alloc(std::size_t size)
{
    return allocator_.allocate(size, pool_.agent_);
}

void hsa_mem_mgr::free(void *ptr)
{
    allocator_.free(ptr);
}

void hsa_mem_mgr::free_device_memory(void *ptr)
{
    hsa_amd_memory_pool_free(ptr);
}


void * hsa_mem_mgr::copy(void *dst, void *src, std::size_t size)
{
    memcpy(dst, src, size);
    return dst;
}

void * hsa_mem_mgr::copy_to_device(void *dst, const void *src, std::size_t size)
{
    hsa_status_t status = hsa_memory_copy(dst, src, size);
    if (status == HSA_STATUS_SUCCESS)
        return dst;
    else
        throw std::exception();
}

void * hsa_mem_mgr::alloc_device_memory(std::size_t size)
{
    void *buffer = NULL;
    size_t mask = pool_.min_alloc_size_ - 1;
    size_t adjusted_size = (size + mask) & mask; // round up allocation to be even numbers of allocation granularity
    hsa_status_t status = hsa_amd_memory_pool_allocate(pool_.pool_, adjusted_size, 0, &buffer);
    if (status != HSA_STATUS_SUCCESS)
        throw std::bad_alloc();
    return buffer;
}

