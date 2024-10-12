#pragma once

#include "utils.h"
#include "dh_comms.h"
#include "inc/comms_mgr.h"

class hsa_mem_mgr : public dh_comms::dh_comms_mem_mgr
{
public:
    hsa_mem_mgr(hsa_agent_t agent,const pool_specs_t& pool, const KernArgAllocator& allocator);
    virtual ~hsa_mem_mgr();
    virtual void * alloc(std::size_t size);
    virtual void free(void *);
    virtual void free_device_memory(void *);
    virtual void * copy(void *dst, void *src, std::size_t size);
    virtual void * alloc_device_memory(std::size_t size);
    virtual void * copy_to_device(void *dst, const void *src, std::size_t size);
private:
    hsa_agent_t agent_;
    const pool_specs_t& pool_;
    const KernArgAllocator& allocator_;

};

