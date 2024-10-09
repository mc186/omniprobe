#include "dh_comms.h"

class hsa_mem_mgr : public dh_comms::dh_comms_mem_mgr
{
public:
    virtual void * alloc(std::size_t size);
    virtual void free(void *);
    virtual void * copy(void *dst, void *src, std::size_t size);
private:

};

