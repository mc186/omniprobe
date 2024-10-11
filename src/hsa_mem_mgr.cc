#include "inc/hsa_mem_mgr.h"


hsa_mem_mgr::hsa_mem_mgr(KernArgAllocator& allocator) : allocator_(allocator)
{
}

void * hsa_mem_mgr::alloc(std::size_t size)
{
    return NULL;
}

void hsa_mem_mgr::free(void *)
{
}

void * hsa_mem_mgr::copy(void *dst, void *src, std::size_t size)
{
    return dst;
}


void * hsa_mem_mgr::alloc_device_memory(std::size_t size)
{
    return NULL;
}

void * hsa_mem_mgr::copy_to_device(void *dst, const void *src, std::size_t size)
{
    return dst;
}
