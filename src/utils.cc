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

#include "inc/utils.h"


using namespace std;
namespace fs = std::filesystem;


std::string demangleName(const char *name)
{
   int status;
   std::string result;
   char *realname = abi::__cxa_demangle(name, 0, 0, &status);
   if (status == 0)
   {
       if (realname)
       {
           result = realname;
           free(realname);
       }
   }
   return result.length() ? result : std::string(name);
}

bool isFileNewer(const std::chrono::system_clock::time_point& timestamp, const std::string& fileName) {
    try {
        // Get the last write time of the file
        auto fileTime = std::filesystem::last_write_time(fileName);
        // Convert the file time to system_clock::time_point
        auto fileTimePoint = std::chrono::time_point_cast<std::chrono::system_clock::duration>(fileTime - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());

        // Compare the file's modification time with the provided timestamp
        return fileTimePoint > timestamp;
    } catch (std::filesystem::filesystem_error& e) {
        // If the file does not exist or another error occurs, return false
        return false;
    }
}

std::string getInstrumentedName(const std::string& func_decl) {
    std::string result = func_decl;
    size_t pos = result.find_last_of(')');
    
    if (pos != std::string::npos) {
        result.replace(pos, 1, ", void*)");
    }
    
    return result;
}


signalPool::signalPool(int initialSize/* = 8 */)
{
}

signalPool::~signalPool()
{
}

hsa_signal_t signalPool::checkout()
{
    lock_guard<mutex> lock(mutex_);
    return {};
}

void signalPool::checkin(hsa_signal_t sig)
{
    lock_guard<mutex> lock(mutex_);
}

coCache::coCache(HsaApiTable *apiTable) : allocator_(apiTable, std::cout)
{
    apiTable_ = apiTable;
}

coCache::~coCache()
{
    lock_guard<std::mutex> lock(mutex);
    for(auto it : cache_objects_)
    {
        CHECK_STATUS("Unable to destroy loaded code objects", apiTable_->core_->hsa_executable_destroy_fn(it.second.executable_));
    }
}

bool coCache::hasKernels(hsa_agent_t agent)
{
    lock_guard<std::mutex> lock(mutex);
    return lookup_map_.find(agent) != lookup_map_.end();
    
}
    
bool coCache::setLocation(hsa_agent_t agent, const std::string& directory, bool instrumented)
{
    filelist_.clear();
    if (directory.length())
    {
        location_ = directory;
        try {
            for (const auto& entry : fs::directory_iterator(directory)) {
                if (entry.is_regular_file() && entry.path().extension() == ".hsaco") {
                    filelist_.push_back(entry.path().string());
                }
            }

            for (auto file : filelist_)
            {
                cout << "coCache::filelist_: " << file << std::endl;
            
                hsa_status_t status = HSA_STATUS_ERROR;

                // Build the code object filename
                std::clog << "Code object filename: " << file << std::endl;

                // Open the file containing code object
                hsa_file_t file_handle = open(file.c_str(), O_RDONLY);
                if (file_handle == -1) {
                    std::cerr << "Error: failed to load '" << file << "'" << std::endl;
                    assert(false);
                    return false;
                }

                // Create code object reader
                hsa_code_object_reader_t code_obj_rdr = {0};
                status = apiTable_->core_->hsa_code_object_reader_create_from_file_fn(file_handle, &code_obj_rdr);
                if (status != HSA_STATUS_SUCCESS) {
                    std::cerr << "Failed to create code object reader '" << file << "'" << std::endl;
                    return false;
                }

                // Create executable.
                hsa_executable_t executable;
                status = apiTable_->core_->hsa_executable_create_alt_fn(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                                     NULL, &executable);
                CHECK_STATUS("Error in creating executable object", status);

                // Load code object.
                status = apiTable_->core_->hsa_executable_load_agent_code_object_fn(executable, agent, code_obj_rdr,
                                                                 NULL, NULL);
                if (status == HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS)
                {
                    cerr << "Looks like " << file << " is not ISA compatible with this GPU\n";
                    apiTable_->core_->hsa_executable_destroy_fn(executable);
                    continue;
                }
                else
                    CHECK_STATUS("Error in loading executable object", status);

                // Freeze executable.
                status = apiTable_->core_->hsa_executable_freeze_fn(executable, "");
                CHECK_STATUS("Error in freezing executable object", status);

                // Get symbol handle.
                hsa_executable_symbol_t kernelSymbol;
                std::vector<hsa_executable_symbol_t> symbols;
                status = apiTable_->core_->hsa_executable_iterate_symbols_fn(executable, [](hsa_executable_t exec, hsa_executable_symbol_t symbol, void *data){
                    std::vector<hsa_executable_symbol_t> *syms = reinterpret_cast<std::vector<hsa_executable_symbol_t> *>(data);
                    syms->push_back(symbol);
                    return HSA_STATUS_SUCCESS;
                }, reinterpret_cast<void *>(&symbols));

                cerr << "coCache found " << symbols.size() << " symbols\n";
                for (auto sym : symbols)
                {
                    hsa_symbol_kind_t kind;
                    CHECK_STATUS("Unable to get valid symbol info", apiTable_->core_->hsa_executable_symbol_get_info_fn(sym,HSA_EXECUTABLE_SYMBOL_INFO_TYPE,&kind));
                    if (kind == HSA_SYMBOL_KIND_KERNEL)
                    {
                        {
                            lock_guard<std::mutex> lock(mutex_);
                            auto it = kernels_.find(agent);
                            if (it != kernels_.end())
                                it->second.push_back(sym);
                            else
                                kernels_[agent].push_back(sym);
                        }

                        uint64_t kernel_object;

						CHECK_STATUS("Can't retrieve a kernel object from a valid symbol", apiTable_->core_->hsa_executable_symbol_get_info_fn(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, reinterpret_cast<void *>(&kernel_object)));
						uint32_t length;
						CHECK_STATUS("Can't retrieve the length of kernel name from a valid symbol", 
							apiTable_->core_->hsa_executable_symbol_get_info_fn(sym, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH,&length));
						char *name = reinterpret_cast<char *>(malloc(length + 1));
						if (name)
						{
							name[length] = '\0';
							CHECK_STATUS("Can't retrieve name from valid symbol", apiTable_->core_->hsa_executable_symbol_get_info_fn(sym, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name));
							string strName = demangleName(name);
                            cerr << "coCache: " << strName << std::endl;
						    free(reinterpret_cast<void *>(name));
                            auto it = lookup_map_.find(agent);
                            if (it != lookup_map_.end())
                                it->second[strName] = sym;
                            else
                                lookup_map_[agent] = {{strName,sym}};
						}
                    }
                }

                {
                    lock_guard<std::mutex> lock(mutex_);
                    cache_objects_[agent] = {executable, file, std::chrono::system_clock::now()};
                }
                close(file_handle);
            }
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Filesystem error: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "General exception: " << e.what() << std::endl;
        }
    }
    return filelist_.size() != 0;
}
    
uint64_t coCache::findAlternative(hsa_executable_symbol_t symbol, const std::string& name)
{
    uint64_t object = 0;
    lock_guard<std::mutex> lock(mutex_);
    hsa_agent_t agent;
    uint32_t kernarg_size;
    CHECK_STATUS("Unable to identify agent for symbol", hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_AGENT, reinterpret_cast<void *>(&agent)));
    CHECK_STATUS("Unable to get kernarg size", hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, reinterpret_cast<void *>(&kernarg_size)));
    auto it = lookup_map_.find(agent);
    if (it != lookup_map_.end())
    {
        auto kern_it = it->second.find(name);
        if (kern_it != it->second.end())
        {
            uint32_t alt_kernarg_size;
            uint64_t alt_kernel_object;
            CHECK_STATUS("Unable to get kernarg size", hsa_executable_symbol_get_info(kern_it->second, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, reinterpret_cast<void *>(&alt_kernarg_size)));
            CHECK_STATUS("Unable to get kernel_object", hsa_executable_symbol_get_info(kern_it->second, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, reinterpret_cast<void *>(&alt_kernel_object)));
            object = alt_kernel_object;
            cout << "kernarg_size = " << kernarg_size << "\nalt_kernarg_size = " << alt_kernarg_size << "\nInstrumentation buffer size = " << sizeof(INSTRUMENTATION_BUFFER) << std::endl;
        }
    }
    return object;
}

uint64_t coCache::findInstrumentedAlternative(hsa_executable_symbol_t, const std::string& name)
{
    return 0;
}


unsigned int getLogDurConfig(std::map<std::string, std::string>& config) {

    // Read the environment variables
    const char* logDurLogLocation = std::getenv("LOGDUR_LOG_LOCATION");
    const char* logDurKernelCache = std::getenv("LOGDUR_KERNEL_CACHE");
    const char* logDurInstrumented = std::getenv("LOGDUR_INSTRUMENTED");

    // If the environment variables are set, add them to the map
    if (logDurLogLocation) {
        config["LOGDUR_LOG_LOCATION"] = std::string(logDurLogLocation);
    } else {
        config["LOGDUR_LOG_LOCATION"] = "console";  // Default or empty value if not set
    }

    if (logDurKernelCache) {
        config["LOGDUR_KERNEL_CACHE"] = std::string(logDurKernelCache);
    } else {
        config["LOGDUR_KERNEL_CACHE"] = "";  // Default or empty value if not set
    }

    if (logDurInstrumented) {
        config["LOGDUR_INSTRUMENTED"] = "true";
    }else {
        config["LOGDUR_INSTRUMENTED"] = "false";
    }

    return config.size();
}

logDuration::logDuration()
{
    location_ = "console";
    if (location_ == "console")
        log_file_ = &cout;
    else
        log_file_ = new std::ofstream(location_, std::ios::app);
    (*log_file_) << "kernel,dispatch,startNs,endNs" << std::endl;
}

logDuration::logDuration(std::string& location)
{
    location_ = location;
    if (location == "console")
        log_file_ = &cout;
    else
        log_file_ = new std::ofstream(location, std::ios::app);
    *log_file_ << "kernel,dispatch,startNs,endNs" << std::endl;
}

logDuration::~logDuration()
{
    if (location_ != "console")
    {
        delete log_file_;
    }
}
    
void logDuration::log(std::string& kernelName, uint64_t dispatchTime, uint64_t startNs, uint64_t endNs)
{
    if (log_file_)
        *log_file_ << kernelName << "," << std::dec << dispatchTime << "," << startNs << "," << endNs << std::endl;
    else
        cerr << "Can't find anyplace to log\n";
}

bool logDuration::setLocation(const std::string& strLocation)
{
    if (location_ != "console" && location_ != "/dev/null")
    {
        if (log_file_)
            delete log_file_;
    }
    cerr << "logDuration::setLocation = " << strLocation << std::endl;
    location_ = strLocation;
    if (!location_.length())
        location_ = "/dev/null";
    if (location_ == "console")
        log_file_ = &cout;
    else
        log_file_ = new std::ofstream(location_, std::ios::app);
    *log_file_ << "kernel,dispatch,startNs,endNs" << std::endl;
    return log_file_ != NULL;
}

void clipKernelName(std::string& str)
{
    // Find the position of the last comma in the string
    size_t lastCommaPos = str.rfind(')');

    // If a comma is found, replace it with a right parenthesis and terminate the string
    if (lastCommaPos != std::string::npos) {
        str.resize(lastCommaPos + 1); // Resize the string to terminate after the parenthesis
    }
}

void clipInstrumentedKernelName(std::string& str) {
    // Find the position of the last comma in the string
    size_t lastCommaPos = str.rfind(',');

    // If a comma is found, replace it with a right parenthesis and terminate the string
    if (lastCommaPos != std::string::npos) {
        str[lastCommaPos] = ')';
        str.resize(lastCommaPos + 1); // Resize the string to terminate after the parenthesis
    }
}


KernArgAllocator::KernArgAllocator(HsaApiTable *apiTable, ostream& out) : out_(out), apiTable_(apiTable)
{
    apiTable_->core_->hsa_iterate_agents_fn([](hsa_agent_t agent, void *data){
        hsa_device_type_t type;
        if (hsa_agent_get_info(agent,HSA_AGENT_INFO_DEVICE, &type) == HSA_STATUS_SUCCESS && type == HSA_DEVICE_TYPE_CPU)
        { 
            hsa_amd_agent_iterate_memory_pools(agent, [](hsa_amd_memory_pool_t pool, void *data){
                hsa_amd_segment_t segment;
                uint32_t flags;
                hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
                if (HSA_AMD_SEGMENT_GLOBAL != segment)
                    return HSA_STATUS_SUCCESS;
                hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
                if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT)
                    reinterpret_cast<KernArgAllocator *>(data)->setPool(pool);
                return HSA_STATUS_SUCCESS;
            }, data);
        }
        return HSA_STATUS_SUCCESS;
    }, this);
}

void * KernArgAllocator::allocate(size_t size, hsa_agent_t allowed)
{
  uint8_t* buffer = NULL;
  size = (size + RH_PAGE_MASK) & ~RH_PAGE_MASK;
  auto status = hsa_amd_memory_pool_allocate(pool_, size, 0, reinterpret_cast<void**>(&buffer));
  // Both the CPU and GPU can access the memory
  if (status == HSA_STATUS_SUCCESS) {
    status = hsa_amd_agents_allow_access(1, &allowed, NULL, buffer);
  }
  uint8_t* ptr = (status == HSA_STATUS_SUCCESS) ? buffer : NULL;
  return ptr;
}

void KernArgAllocator::free(void *ptr)
{
    hsa_amd_memory_pool_free(ptr);
}

KernArgAllocator::~KernArgAllocator()
{
}
    

