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


coCache::~coCache()
{
}
    
bool coCache::setLocation(hsa_agent_t, const std::string& directory, bool instrumented)
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
            
                /*hsa_status_t status = HSA_STATUS_ERROR;

                // Build the code object filename
                std::string filename(brig_path);
                std::clog << "Code object filename: " << filename << std::endl;

                // Open the file containing code object
                hsa_file_t file_handle = open(filename.c_str(), O_RDONLY);
                if (file_handle == -1) {
                    std::cerr << "Error: failed to load '" << filename << "'" << std::endl;
                    assert(false);
                    return false;
                }

                // Create code object reader
                hsa_code_object_reader_t code_obj_rdr = {0};
                status = apiTable_->core_->hsa_code_object_reader_create_from_file(file_handle, &code_obj_rdr);
                if (status != HSA_STATUS_SUCCESS) {
                    std::cerr << "Failed to create code object reader '" << filename << "'" << std::endl;
                    return false;
                }

                // Create executable.
                status = apiTable_->core_->hsa_executable_create_alt_fn(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                                     NULL, executable);
                CHECK_STATUS("Error in creating executable object", status);

                // Load code object.
                status = apiTable_->core_->hsa_executable_load_agent_code_object_fn(*executable, agent_info->dev_id, code_obj_rdr,
                                                                 NULL, NULL);
                CHECK_STATUS("Error in loading executable object", status);

                // Freeze executable.
                status = apiTable_->core_->hsa_executable_freeze_fn(*executable, "");
                CHECK_STATUS("Error in freezing executable object", status);

                // Get symbol handle.
                hsa_executable_symbol_t kernelSymbol;
                status = apiTable_->core_->hsa_executable_get_symbol_fn(*executable, NULL, kernel_name, agent_info->dev_id, 0,
                                                 &kernelSymbol);
                CHECK_STATUS("Error in looking up kernel symbol", status);

                close(file_handle);*/
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
    return 0;
}

uint64_t coCache::findInstrumentedAlternative(hsa_executable_symbol_t, const std::string& name)
{
    return 0;
}


unsigned int getLogDurConfig(std::map<std::string, std::string>& config) {

    // Read the environment variables
    const char* logDurLogLocation = std::getenv("LOGDUR_LOG_LOCATION");
    const char* logDurKernelCache = std::getenv("LOGDUR_KERNEL_CACHE");

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
