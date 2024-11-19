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
#include <unordered_set>
namespace kernelDB {

typedef struct instruction_s{
    std::string prefix_;
    std::string type_;
    std::string size_;
    std::string inst_;
    std::vector<std::string> operands_;
    std::string disassembly_;
}instruction_t;


enum parse_mode {
    BEGIN,
    KERNEL,
    BBLOCK,
    BRANCH
};


class basicBlock {
public: 
    basicBlock(uint16_t id);
    ~basicBlock() = default;
    void addInstruction(const instruction_t& instruction);
private:
    uint16_t block_id;
    std::string disassembly_;
    std::vector<instruction_t> instructions_;
    std::map<std::string, uint64_t> counts_;
};

class CDNAKernel {
public:
    CDNAKernel(const std::string& name, const std::string& disassembly);
    ~CDNAKernel() = default;
private:
    std::string name_;
    std::string disassembly_;
    std::vector<basicBlock> blocks_;
};

class __attribute__((visibility("default"))) kernelDB {
public:
    kernelDB(hsa_agent_t agent, const std::string& fileName);
    kernelDB(hsa_agent_t agent, std::vector<uint8_t> bits);
    ~kernelDB();
    bool getBasicBlocks(const std::string& name, std::vector<basicBlock>&);
    bool addFile(const std::string& name, hsa_agent_t agent, const std::string& strFilter);
    bool parseDisassembly(const std::string& text);
    static void getElfSectionBits(const std::string &fileName, const std::string &sectionName, std::vector<uint8_t>& sectionData );
private:
    parse_mode getLineType(std::string& line);
    static bool isBranch(const std::string& instruction);
private:
    std::map<std::string, CDNAKernel> kernels_;
    amd_comgr_data_t executable_;
    hsa_agent_t agent_;
    std::string fileName_;
};


}//kernelDB
