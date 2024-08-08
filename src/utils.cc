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

coCache::coCache(std::string& directory)
{
    try {
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_regular_file() && entry.path().extension() == ".hsaco") {
                filelist_.push_back(entry.path().string());
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
				status = hsa_api_.hsa_code_object_reader_create_from_file(file_handle, &code_obj_rdr);
				if (status != HSA_STATUS_SUCCESS) {
					std::cerr << "Failed to create code object reader '" << filename << "'" << std::endl;
					return false;
				}

				// Create executable.
				status = hsa_api_.hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
													 NULL, executable);
				CHECK_STATUS("Error in creating executable object", status);

				// Load code object.
				status = hsa_api_.hsa_executable_load_agent_code_object(*executable, agent_info->dev_id, code_obj_rdr,
																 NULL, NULL);
				CHECK_STATUS("Error in loading executable object", status);

				// Freeze executable.
				status = hsa_api_.hsa_executable_freeze(*executable, "");
				CHECK_STATUS("Error in freezing executable object", status);

				// Get symbol handle.
				hsa_executable_symbol_t kernelSymbol;
				status = hsa_api_.hsa_executable_get_symbol(*executable, NULL, kernel_name, agent_info->dev_id, 0,
												 &kernelSymbol);
				CHECK_STATUS("Error in looking up kernel symbol", status);

				close(file_handle);*/
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "General exception: " << e.what() << std::endl;
    }
}

coCache::~coCache()
{
}

hsa_executable_t coCache::getInstrumented(hsa_executable_t, std::string name)
{
    lock_guard<mutex> lock(mutex_);
    return {};
}


