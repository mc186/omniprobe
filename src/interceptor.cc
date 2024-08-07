

/******************************************************************************
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*******************************************************************************/

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Test tool used to experiment with interactive profiling.                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
#undef NDEBUG
#include <assert.h>
#include <cxxabi.h>
#include <dirent.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>   /* For SYS_xxx definitions */
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <limits.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <fstream>
#include <list>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <thread>
#include <mutex>
#include <utility>
#include <shared_mutex>
#include <filesystem>
#include <ios>
#include <ctime>
#include <algorithm>
#include <regex>
#include <fcntl.h>
#include <sys/stat.h>


#include "inc/interceptor.h"


using namespace std;

int PID = getpid();

ofstream LOG(string("/tmp") + "/rocscope.log." + to_string(PID));

template<typename T>
void PR(const T& arg) { LOG << arg << '\n' << flush; }

template<typename T, typename ...TS>
void PR(const T& arg, const TS&... args) { LOG << arg << " "; PR(args...); }

const char *DBG = ">>>>>>>>";
const char *INTERCEPTOR_MSG = "[hsaInterceptor] ";

hsaInterceptor * hsaInterceptor::getInstance(HsaApiTable *table, uint64_t runtime_version, uint64_t failed_tool_count, const char* const* failed_tool_names)
{
    const lock_guard<mutex> lock(mutex_);
    if (!singleton_)
    {
        if (table != NULL)
        {
            singleton_ = new hsaInterceptor(table, runtime_version, failed_tool_count, failed_tool_names);
            singleton_->saveHsaApi();
            singleton_->hookApi();
        }
        else
            cerr << "hsaInterceptor Initialization failed - API table is NULL" << endl;
    }
    return singleton_;
}

void hsaInterceptor::OnSubmitPackets(const void* in_packets, uint64_t count,
    uint64_t user_que_idx, void* data, hsa_amd_queue_intercept_packet_writer writer)
{
    hsaInterceptor *hook = hsaInterceptor::getInstance();
    if (hook)
    {
        hsa_queue_t *queue = reinterpret_cast<hsa_queue_t *>(data);
        hook->doPackets(queue, static_cast<const packet_t *>(in_packets), count, writer);
        cout << "Enqueued " << count << " packet(s)\n";
    }
}


hsaInterceptor::hsaInterceptor(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count, const char* const* failed_tool_names) {
    apiTable_ = table;
}
hsaInterceptor::~hsaInterceptor() {
}

decltype(hsa_queue_create)* hsa_queue_create_fn;
decltype(hsa_queue_destroy)* hsa_queue_destroy_fn;
decltype(hsa_amd_queue_intercept_create)* hsa_amd_queue_intercept_create_fn;
decltype(hsa_amd_queue_intercept_register)* hsa_amd_queue_intercept_register_fn;

void hsaInterceptor::saveHsaApi() {


  hsa_queue_create_fn = apiTable_->core_->hsa_queue_create_fn;
  hsa_queue_destroy_fn = apiTable_->core_->hsa_queue_destroy_fn;
  hsa_amd_queue_intercept_create_fn = apiTable_->amd_ext_->hsa_amd_queue_intercept_create_fn;
  hsa_amd_queue_intercept_register_fn = apiTable_->amd_ext_->hsa_amd_queue_intercept_register_fn;
}

void hsaInterceptor::restoreHsaApi() {

  apiTable_->core_->hsa_queue_create_fn = hsa_queue_create_fn;
  apiTable_->core_->hsa_queue_destroy_fn = hsa_queue_destroy_fn;
  apiTable_->amd_ext_->hsa_amd_queue_intercept_create_fn = hsa_amd_queue_intercept_create_fn;
  apiTable_->amd_ext_->hsa_amd_queue_intercept_register_fn = hsa_amd_queue_intercept_register_fn;

}


string hsaInterceptor::packetToText(const packet_t *packet)
{
    assert(packet && "packet null!");
    ostringstream buff;
    uint32_t type = getHeaderType(packet);
    if (type == HSA_PACKET_TYPE_KERNEL_DISPATCH)
    {
        const hsa_kernel_dispatch_packet_t *disp = reinterpret_cast<const hsa_kernel_dispatch_packet_t *>(packet);
        uint32_t scope = getHeaderReleaseScope(disp);
        buff << "Dispatch Packet\n";
        buff << "\tRelease Scope: ";
        if (scope & HSA_FENCE_SCOPE_AGENT)
            buff << "HSA_FENCE_SCOPE_AGENT";
        else if (scope & HSA_FENCE_SCOPE_SYSTEM)
            buff << "HSA_FENCE_SCOPE_SYSTEM";
        else
            buff << hex << scope;
        buff << "\n";
        buff << "\tAcquire Scope: ";
        scope = getHeaderAcquireScope(disp);
        if (scope & HSA_FENCE_SCOPE_AGENT)
            buff << "HSA_FENCE_SCOPE_AGENT";
        else if (scope & HSA_FENCE_SCOPE_SYSTEM)
            buff << "HSA_FENCE_SCOPE_SYSTEM";
        else
            buff << "UNKNOWN";
        buff << "\n";
        buff << "\tsetup: 0x" << hex << disp->setup << "\n";
        buff << "\tworkgroup_size_x: 0x" << hex << disp->workgroup_size_x << "\n";
        buff << "\tworkgroup_size_y: 0x" << hex << disp->workgroup_size_y << "\n";
        buff << "\tworkgroup_size_z: 0x" << hex << disp->workgroup_size_z << "\n";
        buff << "\tgrid_size_x: 0x" << hex << disp->grid_size_x << "\n";
        buff << "\tgrid_size_y: 0x" << hex << disp->grid_size_y << "\n";
        buff << "\tgrid_size_z: 0x" << hex << disp->grid_size_z << "\n";
        buff << "\tprivate_segment_size: 0x" << hex << disp->private_segment_size << "\n";
        buff << "\tgroup_segment_size: 0x" << hex << disp->group_segment_size << "\n";
        buff << "\tkernel_object: 0x" << hex << disp->kernel_object << "\n";
        buff << "\tkernarg_address: 0x" << hex << disp->kernarg_address << "\n";
        buff << "\tcompletion_signal 0x" << hex << disp->completion_signal.handle << "\n";
    }
    else
    {
        buff << "Unsupported packet type\n";
    }
    return buff.str();
}

/*
    This is the packet handler registered with the intercept queue created by hsa_queue_create(...)
    In this method we inspect every packet, writing every non-dispatch packet to the destination queue.
    But for dispatch packets, we delegate decision making to either runKernel, timeKernel, or traceKernel.  runKernel
    gathers hardware counters while traceKernel gathers SQTT data. timeKernel merely collects timing
    data on the kernel but doesn't otherwise interfere with concurrent execution.  Both runKernel and traceKernel
    take a unique_lock to guarantee serialized kernel execution (required because hardware can't differentiate
    counter values between concurrently running kernels.)
*/
void hsaInterceptor::doPackets(hsa_queue_t *queue, const packet_t *packet, uint64_t count, hsa_amd_queue_intercept_packet_writer writer) {
    try {

       writer(packet, count); 

    } catch(const exception& e) {
        debug_out(cerr, INTERCEPTOR_MSG, e.what());
    }
}



void hsaInterceptor::addQueue(hsa_queue_t *queue, hsa_agent_t agent)
{
    //debug_out(cout, "ROCMHOOK: addQueue 0x", hex, queue->doorbell_signal.handle, " tid: ", gettid());
    lock_guard<mutex> lock(mm_mutex_);

    auto result = (*(apiTable_->amd_ext_->hsa_amd_profiling_set_profiler_enabled_fn))(queue, true);

    assert(result == HSA_STATUS_SUCCESS && "Couldn't enable queue for profiling");

}


/*
    This is really the key to wiring ourselves into the HSA queuing pipeline.  This static method is written into
    the HsaApiTable at OnLoad time so that application calls to hsa_queue_create (e.g. by the HIP runtime) get resolved
    to this method.  The key here is to not merely forward the call to the original hsa_queue_create API but to create
    an alternative TYPE of queue - an "intercept" queue.  This is a functionality implemented by the HSA runtime
    for the purpose of giving callers control over the AQL packet pipeline.  So we create an intercept queue via
    hsa_amd_queue_intercept_create, and then register a handler for packets enqueued to that particular queue - see the
    call to hsa_amd_queue_intercept_register. By doing this we get invoked on every AQL packet dispatch.  This gives
    us the ability to wrap, as needed, dispatch packets in the PM4 packets needed to start and stop counter and trace collection.
*/
hsa_status_t hsaInterceptor::hsa_queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
    void(*callback)(hsa_status_t status, hsa_queue_t *source, void *data),
    void *data, uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t **queue) {
    hsa_status_t result = HSA_STATUS_SUCCESS;
    try {
        result = (*hsa_amd_queue_intercept_create_fn)(agent, size, type, callback, data, private_segment_size, group_segment_size, queue);
        if (result == HSA_STATUS_SUCCESS) {
            hsaInterceptor *instance = getInstance();
            instance->addQueue(*queue, agent);
            auto hookResult = (*hsa_amd_queue_intercept_register_fn)(*queue, hsaInterceptor::OnSubmitPackets, reinterpret_cast<void *>(*queue));
            if (hookResult != HSA_STATUS_SUCCESS) {
                debug_out(cerr, "hsaInterceptor: Failed to register intercept callback with result of ", hookResult);
            } else {
                cout << DBG << " HSA intercept registered.\n";
            }
        }
    } catch(const exception& e) {
        cerr << e.what();
    }
    return result;
}

hsa_status_t hsaInterceptor::hsa_queue_destroy(hsa_queue_t *queue)
{
    hsa_status_t result = HSA_STATUS_SUCCESS;
    try
    {
        result = (*hsa_queue_destroy_fn)(queue);
    }
    catch(const exception& e)
    {
        cerr << e.what();
    }
    return result;
}

void hsaInterceptor::hookApi(){
    apiTable_->core_->hsa_queue_create_fn = hsaInterceptor::hsa_queue_create;
    apiTable_->core_->hsa_queue_destroy_fn = hsaInterceptor::hsa_queue_destroy;

}

hsaInterceptor * hsaInterceptor::singleton_ = NULL;
mutex hsaInterceptor::mutex_;
shared_mutex hsaInterceptor::stop_mutex_;

timeHelper globalTime;

extern "C" {

    /*
        The HSA runtime has a feature for which you can write extension libraries to hook out the HSA runtime
        API to do clever things like profilers.  The environment variable HSA_TOOLS_LIB points to a shared
        lib that contains an "OnLoad" method which will be invoked by the HSA runtime.  One of the method
        parameters is what amounts to a vtable of the entire HSA api.  Overwriting the vtable entries
        allows the HSA_TOOL to masquerade as the runtime to gain visibility on the application itself.
        Typically an HSA_TOOL will forward calls on to the original APIs once it has processed the call beforehand.

        rocmhook constructs a view of the application (memory allocations, kernels loaded, etc) and uses that
        to support the ability to capture traces and hardware counters during kernel execution. rocmhook also has
        the ability to take snapshots of device memory and restore the contents of memory during runtime.  This was
        implemented to support kernel replay, which does not occur by default but must be explicitly invoked.

    */

    PUBLIC_API bool OnLoad(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count,
                           const char* const* failed_tool_names) {
        cout << "Trying to init hsaInterceptor" << endl;
        hsaInterceptor *hook = hsaInterceptor::getInstance(table, runtime_version, failed_tool_count, failed_tool_names);
        cout << "hsaInterceptor: Initializing: 0x" << hex << hook << endl;

        return true;
    }

    PUBLIC_API void OnUnload() {
        // cout << "ROCMHOOK: Unloading" << endl;
        //hsaInterceptor::cleanup();
        printf("Elapsed usecs: %lu\n", globalTime.getElapsedNanos() / 1000);
    }

    static void unload_me() __attribute__((destructor));
    void unload_me()
    {
        // cout << "ROCMHOOK: Unloading" << endl;
        //hsaInterceptor::cleanup();
        printf("hsaInterceptor: Application elapsed usecs: %lu\n", globalTime.getElapsedNanos() / 1000);
    }
}
