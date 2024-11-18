/******************************************************************************
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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
#include "utils.h"
namespace kernelDB {
typedef struct basicBlock_s {
    std::string disassembly_;
    std::map<std::string, uint64_t> counts_;
}basicBlock_t;

typedef struct kernel_s {
    std::string name_;
    std::string disassembly_;
    std::vector<basicBlock_t> blocks_;
}kernel_t;

class __attribute__((visibility("default"))) kernelDB {
public:
    kernelDB(hsa_agent_t agent, const std::string& fileName);
    kernelDB(hsa_agent_t agent, std::vector<uint8_t> bits);
    ~kernelDB();
    bool getBasicBlocks(const std::string& name, std::vector<basicBlock_t>&);
    bool addFile(const std::string& name, hsa_agent_t agent, const std::string& strFilter);
    bool parseDisassembly(const std::string& text);
    static void getElfSectionBits(const std::string &fileName, const std::string &sectionName, std::vector<uint8_t>& sectionData );
private:
    std::map<std::string, kernel_t> kernels_;
    amd_comgr_data_t executable_;
    hsa_agent_t agent_;
    std::string fileName_;
};

enum parse_mode {
    BEGIN,
    KERNEL,
    BBLOCK,
    INSTRUCTION
};

}//kernelDB
