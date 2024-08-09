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



using namespace std;

#define CHECK_STATUS(msg, status) do {                                                             \
  if ((status) != HSA_STATUS_SUCCESS) {                                                            \
    const char* emsg = 0;                                                                          \
    hsa_status_string(status, &emsg);                                                              \
    printf("%s: %s\n", msg, emsg ? emsg : "<unknown error>");                                      \
    abort();                                                                                       \
  }                                                                                                \
} while (0)


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

class coCache{
public:
    coCache(HsaApiTable *apiTable) {apiTable_ = apiTable;}
    coCache(std::string& directory);
    ~coCache();
    bool setLocation(const std::string& directory);
    hsa_executable_t getInstrumented(hsa_executable_t, std::string name);
private:
    HsaApiTable *apiTable_;
    std::map<std::string, symbol_info_t> symbol_cache_;
    std::map<std::string,hsa_executable_t> cache_;
    std::vector<std::string> filelist_;
    std::mutex mutex_;
    std::string location_;
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
