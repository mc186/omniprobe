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
#pragma once
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

#include <hsa.h>
#include <hsa_ven_amd_aqlprofile.h>
#include <hsa_ven_amd_loader.h>
#include "rocprofiler/rocprofiler.h"
#define AMD_INTERNAL_BUILD
#include <hsa_api_trace.h>


#define RH_PAGE_SIZE 0x1000
#define RH_PAGE_MASK 0x0FFF

using namespace std;

#define CHECK_STATUS(msg, status) do {                                                             \
  if ((status) != HSA_STATUS_SUCCESS) {                                                            \
    const char* emsg = 0;                                                                          \
    hsa_status_string(status, &emsg);                                                              \
    printf("%s: %s\n", msg, emsg ? emsg : "<unknown error>");                                      \
    abort();                                                                                       \
  }                                                                                                \
} while (0)

template<typename T>
struct hsa_cmp
{
    bool operator() (const T& first, const T& second) const
    {
        return first.handle < second.handle;
    }
};

class KernArgAllocator
{
public:
    KernArgAllocator(HsaApiTable *pTable, std::ostream& out);
    ~KernArgAllocator();
    void *allocate(size_t size, hsa_agent_t allowed);
    void *allocate(size_t size);
    void free(void *ptr);
    void setPool(hsa_amd_memory_pool_t pool) {pool_ = pool;}
    std::ostream& out_;
private:
    hsa_amd_memory_pool_t pool_;
    HsaApiTable *apiTable_;
    hsa_agent_t agent;
};

class signalPool{
public:
    signalPool(int initialSize = 8);
    ~signalPool();
    hsa_signal_t checkout();
    void checkin(hsa_signal_t sig);
private:
    std::vector<hsa_signal_t> available_;
    std::vector<hsa_signal_t> in_use_;
    std::mutex mutex_;
};


typedef struct symbol_info {
}symbol_info_t;

typedef struct cache_object{
    hsa_executable_t executable_;
    std::string filename_;
    std::chrono::time_point<std::chrono::system_clock> timestamp_;
}cache_object_t;

class coCache{
public:
    coCache(HsaApiTable *apiTable) {apiTable_ = apiTable;}
    ~coCache();
    bool setLocation(hsa_agent_t agent, const std::string& directory, bool instrumented = true);
    uint64_t findAlternative(hsa_executable_symbol_t symbol, const std::string& name);
    uint64_t findInstrumentedAlternative(hsa_executable_symbol_t, const std::string& name);
    bool hasKernels();
private:
    HsaApiTable *apiTable_;
    std::map<std::string, symbol_info_t> symbol_cache_;
    std::map<std::string,hsa_executable_t> cache_;
    std::map<hsa_agent_t, std::vector<hsa_executable_symbol_t>, hsa_cmp<hsa_agent_t>> kernels_;
    std::vector<std::string> filelist_;
    std::mutex mutex_;
    std::string location_;
    std::map<hsa_agent_t, cache_object_t, hsa_cmp<hsa_agent_t>> cache_objects_;
};

class logDuration{
public:
    logDuration();
    logDuration(std::string& location);
    ~logDuration();
    void log(std::string& kernelName, uint64_t dispatchTime, uint64_t startNs, uint64_t endNs);
    bool setLocation(const std::string& strLocation);
private:
    std::ostream *log_file_;
    std::string location_;
};


unsigned int getLogDurConfig(std::map<std::string, std::string>& config);
void clipInstrumentedKernelName(std::string& str);
void clipKernelName(std::string& str);
bool isFileNewer(const std::chrono::system_clock::time_point& timestamp, const std::string& fileName);
