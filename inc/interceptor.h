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


class hsaInterceptor;

#define PUBLIC_API __attribute__((visibility("default")))
#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API __attribute__((destructor))

#define REPLAY_QUEUE_SIZE 512

#ifndef NDEBUG
	template<typename ...Args>
	void debug_out(std::ostream& out, Args && ...args)
	{
		(out << ... << args);
        out << std::endl;
	}
#else
    #define debug_out(...)
#endif


typedef uint32_t packet_word_t;
typedef hsa_kernel_dispatch_packet_t dispatch_packet_t;
typedef hsa_ext_amd_aql_pm4_packet_t packet_t;



template<typename T>
struct hsa_cmp
{
    bool operator() (const T& first, const T& second) const
    {
        return first.handle < second.handle;
    }
};


static const int CHECKSUM_PAGE_SIZE = 1 << 20;


class hsaInterceptor {
private:
    hsaInterceptor(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count, const char* const* failed_tool_names);
    virtual ~hsaInterceptor();
    void restoreHsaApi();
    void saveHsaApi();
    void hookApi();
    void addQueue(hsa_queue_t *queue, hsa_agent_t agent);
    void removeQueue(hsa_queue_t *queue);
    static void OnSubmitPackets(const void* in_packets, uint64_t count, uint64_t user_que_idx, void* data,
                         hsa_amd_queue_intercept_packet_writer writer);
    static hsa_status_t hsa_queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type, void(*callback)(hsa_status_t status, hsa_queue_t *source, void *data), void *data, uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t **queue);
    static hsa_status_t hsa_queue_destroy(hsa_queue_t *queue);
    virtual void doPackets(hsa_queue_t *queue, const packet_t *packet, uint64_t count, hsa_amd_queue_intercept_packet_writer writer);

public:
    void addAgent(hsa_agent_t agent, unsigned int dev_index);
    std::string packetToText(const packet_t *packet);
    static hsaInterceptor *getInstance(HsaApiTable *table = NULL, uint64_t runtime_version = 0, uint64_t failed_tool_count = 0, const char* const* failed_tool_names = NULL);
    static void cleanup();
    static const packet_word_t header_type_mask = (1ul << HSA_PACKET_HEADER_WIDTH_TYPE) - 1;
    static const packet_word_t header_screlease_scope_mask = 0x3;
    static const packet_word_t header_scacquire_scope_mask = 0x3;
    static hsa_packet_type_t getHeaderType(const packet_t* packet) {
        const packet_word_t* header = reinterpret_cast<const packet_word_t*>(packet);
        return static_cast<hsa_packet_type_t>((*header >> HSA_PACKET_HEADER_TYPE) & header_type_mask);
    }
    static hsa_packet_type_t getHeaderReleaseScope(const dispatch_packet_t* packet) {
        const packet_word_t* header = reinterpret_cast<const packet_word_t*>(packet);
        return static_cast<hsa_packet_type_t>((*header >> HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE) & header_screlease_scope_mask);
    }
    static hsa_packet_type_t getHeaderAcquireScope(const dispatch_packet_t* packet) {
        const packet_word_t* header = reinterpret_cast<const packet_word_t*>(packet);
        return static_cast<hsa_packet_type_t>((*header >> HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) & header_scacquire_scope_mask);
    }
private:
    HsaApiTable *apiTable_;
    std::map<hsa_queue_t *, std::pair<unsigned int, uint64_t>> queue_ids_;
    std::map<std::string, hsa_agent_t> agents_;
    static std::mutex mutex_;
    static std::shared_mutex stop_mutex_;
    std::mutex mm_mutex_;
    static hsaInterceptor *singleton_;
};


extern "C" {
    PUBLIC_API bool OnLoad(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count,
                           const char* const* failed_tool_names);
    PUBLIC_API void OnUnload();
}
