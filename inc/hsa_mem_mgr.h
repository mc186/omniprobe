#pragma once

#include "utils.h"
#include "dh_comms.h"

class hsa_mem_mgr : public dh_comms::dh_comms_mem_mgr
{
public:
    hsa_mem_mgr(KernArgAllocator& allocator);
    virtual void * alloc(std::size_t size);
    virtual void free(void *);
    virtual void * copy(void *dst, void *src, std::size_t size);
    virtual void * alloc_device_memory(std::size_t size);
    virtual void * copy_to_device(void *dst, const void *src, std::size_t size);
private:
    KernArgAllocator& allocator_;
};

