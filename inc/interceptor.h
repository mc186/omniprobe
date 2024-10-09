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

class hsaInterceptor;
void signal_runner();
void cache_watcher();

#define PUBLIC_API __attribute__((visibility("default")))
#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API __attribute__((destructor))

#define REPLAY_QUEUE_SIZE 512
#define SIGPOOL_INCREMENT 8

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




static const int CHECKSUM_PAGE_SIZE = 1 << 20;

typedef struct kernel_info{
    hsa_signal_t signal_;
    std::string name_;
    hsa_agent_t agent_;
    timeHelper th_;
}kernel_info_t;

typedef struct ld_kernel_descriptor {
    std::string name_;
    hsa_executable_symbol_t symbol_;
    hsa_agent_t agent_;
    uint32_t kernarg_size_;
}ld_kernel_descriptor_t;

class hsaInterceptor {
private:
    hsaInterceptor(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count, const char* const* failed_tool_names);
    virtual ~hsaInterceptor();
    void restoreHsaApi();
    void saveHsaApi();
    void hookApi();
    void addQueue(hsa_queue_t *queue, hsa_agent_t agent);
    void removeQueue(hsa_queue_t *queue);
    void addKernel(uint64_t kernelObject, std::string& name, hsa_executable_symbol_t symbol, hsa_agent_t agent, uint32_t kernarg_size);
    bool getPendingSignals(std::vector<hsa_signal_t>& outSigs);
    void signalCompleted(const hsa_signal_t sig);
    bool signalWait(hsa_signal_t sig, uint64_t timeout);
    static void OnSubmitPackets(const void* in_packets, uint64_t count, uint64_t user_que_idx, void* data,
                         hsa_amd_queue_intercept_packet_writer writer);
    static hsa_status_t hsa_queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type, void(*callback)(hsa_status_t status, hsa_queue_t *source, void *data), void *data, uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t **queue);
    static hsa_status_t hsa_queue_destroy(hsa_queue_t *queue);
    static hsa_status_t hsa_executable_symbol_get_info(hsa_executable_symbol_t symbol, hsa_executable_symbol_info_t attribute, void *data);
    hsa_kernel_dispatch_packet_t *fixupPacket(const hsa_kernel_dispatch_packet_t *packet, hsa_queue_t *queue);
    virtual void doPackets(hsa_queue_t *queue, const packet_t *packet, uint64_t count, hsa_amd_queue_intercept_packet_writer writer);

public:
    void addAgent(hsa_agent_t agent, unsigned int dev_index);
    std::string packetToText(const packet_t *packet);
    static hsaInterceptor *getInstance(HsaApiTable *table = NULL, uint64_t runtime_version = 0, uint64_t failed_tool_count = 0, const char* const* failed_tool_names = NULL);
    static void cleanup();
    static hsa_packet_type_t getHeaderType(const packet_t* packet) {
        const packet_word_t* header = reinterpret_cast<const packet_word_t*>(packet);
        return static_cast<hsa_packet_type_t>((*header >> HSA_PACKET_HEADER_TYPE) & header_type_mask);
    }
    static const packet_word_t header_type_mask = (1ul << HSA_PACKET_HEADER_WIDTH_TYPE) - 1;
    static const packet_word_t header_screlease_scope_mask = 0x3;
    static const packet_word_t header_scacquire_scope_mask = 0x3;
    static hsa_packet_type_t getHeaderReleaseScope(const dispatch_packet_t* packet) {
        const packet_word_t* header = reinterpret_cast<const packet_word_t*>(packet);
        return static_cast<hsa_packet_type_t>((*header >> HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE) & header_screlease_scope_mask);
    }
    static hsa_packet_type_t getHeaderAcquireScope(const dispatch_packet_t* packet) {
        const packet_word_t* header = reinterpret_cast<const packet_word_t*>(packet);
        return static_cast<hsa_packet_type_t>((*header >> HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) & header_scacquire_scope_mask);
    }
    friend void signal_runner();
    friend void cache_watcher();
protected:
    bool shuttingdown();
    void shutdown();
    std::string getCacheLocation() { return config_["LOGDUR_KERNEL_CACHE"];}
    bool addCodeObject(const std::string& name);
private:
    HsaApiTable *apiTable_;
    std::map<hsa_queue_t *, std::pair<unsigned int, uint64_t>> queue_ids_;
    std::map<std::string, hsa_agent_t> agents_;
    std::map<hsa_queue_t *, hsa_agent_t> queues_;
    std::map<hsa_agent_t, std::string, hsa_cmp<hsa_agent_t>> isas_;
    std::map<hsa_signal_t, kernel_info_t, hsa_cmp<hsa_signal_t>> pending_signals_;
    std::vector<hsa_signal_t> sig_pool_;
    std::map<uint64_t, ld_kernel_descriptor_t> kernel_objects_;
    std::map<hsa_signal_t, hsa_signal_t, hsa_cmp<hsa_signal_t>> app_sigs_;
    std::vector<dh_comms::dh_comms *> buffers_;
    std::map<std::string, std::string> config_;
    std::map<hsa_signal_t, void *, hsa_cmp<hsa_signal_t>> kernargs_;
    uint64_t dispatch_count_;
    std::atomic<bool> shutting_down_;
    std::thread signal_runner_;
    std::thread cache_watcher_;
    std::mutex mutex_;
    logDuration log_;
    coCache kernel_cache_;
    bool run_instrumented_;
    KernArgAllocator allocator_;
    hsa_mem_mgr mem_mgr_;
    std::map<hsa_signal_t, void *, hsa_cmp<hsa_signal_t>> pending_kernargs_;
    static std::mutex singleton_mutex_;
    static std::shared_mutex stop_mutex_;
    static hsaInterceptor *singleton_;
};


extern "C" {
    PUBLIC_API bool OnLoad(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count,
                           const char* const* failed_tool_names);
    PUBLIC_API void OnUnload();
}
