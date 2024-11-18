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

#include <iostream>
#include <sstream>
#include "inc/kernelDB.h"
#define FATBIN_SECTION ".hip_fatbin"

#define CHECK_COMGR(call)                                                                          \
  if (amd_comgr_status_s status = call) {                                                          \
    const char* reason = "";                                                                       \
    amd_comgr_status_string(status, &reason);                                                      \
    std::cerr << __LINE__ << " code: " << status << std::endl;                                     \
    std::cerr << __LINE__ << " failed: " << reason << std::endl;                                   \
    exit(1);                                                                                       \
  }


namespace kernelDB {

std::string getExecutablePath() {
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    return std::string(result, (count > 0) ? count : 0);
}

std::vector<std::string> getIsaList(hsa_agent_t agent)
{
    std::vector<std::string> list;
    hsa_agent_iterate_isas(agent,[](hsa_isa_t isa, void *data){
        std::vector<std::string> *pList = reinterpret_cast<std::vector<std::string> *> (data);
           uint32_t length;
           hsa_status_t status = hsa_isa_get_info(isa, HSA_ISA_INFO_NAME_LENGTH, 0, &length);
           if (status == HSA_STATUS_SUCCESS)
           {
                char *pName = static_cast<char *>(malloc(length + 1));
                if (pName)
                {
                    pName[length] = '\0';
                    status = hsa_isa_get_info(isa, HSA_ISA_INFO_NAME, 0, pName);
                    //std::cerr << "Isa name: " << pName << std::endl;
                    if (status == HSA_STATUS_SUCCESS)
                        pList->push_back(std::string(pName));
                    free(pName);
                }
                else
                {
                    std::cout << "The system is somehow out of memory at line " << __LINE__ << " so I'm aborting this run." << std::endl;
                    abort();
                }
           }
           return HSA_STATUS_SUCCESS;
        }, reinterpret_cast<void *>(&list));   
    return list;
}


kernelDB::kernelDB(hsa_agent_t agent, const std::string& fileName)
{
    agent_ = agent;
    fileName_ = fileName;
    std::string empty("");
    addFile(fileName, agent, empty);
}

kernelDB::kernelDB(hsa_agent_t agent, std::vector<uint8_t> bits)
{
}

kernelDB::~kernelDB()
{
   std::cout << "Ending kernelDB\n"; 
}

bool kernelDB::getBasicBlocks(const std::string& kernel, std::vector<basicBlock_t>&)
{
    return true;
}

bool kernelDB::addFile(const std::string& name, hsa_agent_t agent, const std::string& strFilter)
{
    bool bReturn = true;
    amd_comgr_data_t executable;
    std::vector<std::string> isas = ::kernelDB::getIsaList(agent);
    if (name.ends_with(".hsaco"))
    {
        std::vector<char> buff;
        std::ifstream file(name, std::ios::binary | std::ios::ate);

        if (!file.is_open()) {
            std::cerr << "Failed to open the file: " << name << std::endl;
        }

        std::streamsize fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        // Resize the buffer to fit the file content
        buff.resize(fileSize);

        // Read the file content into the buffer
        if (!file.read(buff.data(), fileSize)) {
            std::cerr << "Failed to read the file content" << std::endl;
        }
        file.close();
        CHECK_COMGR(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &executable));
        CHECK_COMGR(amd_comgr_set_data(executable, buff.size(), buff.data()));
    }
    else
    {
        amd_comgr_data_t bundle;
        std::vector<uint8_t> bits;
        getElfSectionBits(name, FATBIN_SECTION, bits);
        CHECK_COMGR(amd_comgr_create_data(AMD_COMGR_DATA_KIND_FATBIN, &bundle));
        CHECK_COMGR(amd_comgr_set_data(bundle, bits.size(), reinterpret_cast<const char *>(bits.data())));
        if (isas.size())
        {
            std::vector<amd_comgr_code_object_info_t> ql;
            for (int i = 0; i < isas.size(); i++)
                ql.push_back({isas[i].c_str(),0,0});
            CHECK_COMGR(amd_comgr_lookup_code_object(bundle,static_cast<amd_comgr_code_object_info_t *>(ql.data()), ql.size()));
            for (auto co : ql)
            {
                /* Use the first code object that is ISA-compatible with this agent */
                if (co.size != 0)
                {
                    CHECK_COMGR(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &executable));
                    CHECK_COMGR(amd_comgr_set_data(executable, co.size, reinterpret_cast<const char *>(bits.data() + co.offset)));
                    break;
                }
            }   
        }
        CHECK_COMGR(amd_comgr_release_data(bundle));
    }
    if(isas.size())
    {
        amd_comgr_data_set_t dataSetIn, dataSetOut;
        amd_comgr_data_t dataOutput;
        amd_comgr_action_info_t dataAction;
        CHECK_COMGR(amd_comgr_create_data_set(&dataSetIn));
        CHECK_COMGR(amd_comgr_set_data_name(executable, "RB_DATAIN"));
        CHECK_COMGR(amd_comgr_data_set_add(dataSetIn, executable));
        CHECK_COMGR(amd_comgr_create_data_set(&dataSetOut));
        CHECK_COMGR(amd_comgr_create_action_info(&dataAction));
        CHECK_COMGR(amd_comgr_action_info_set_isa_name(dataAction,isas[0].c_str()));
    	CHECK_COMGR(amd_comgr_do_action(AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE,
                            dataAction, dataSetIn, dataSetOut));
		CHECK_COMGR(amd_comgr_destroy_data_set(dataSetIn));
		size_t count,size;
        
		CHECK_COMGR(amd_comgr_action_data_count(dataSetOut, AMD_COMGR_DATA_KIND_SOURCE, &count));
		CHECK_COMGR(amd_comgr_action_data_get_data(dataSetOut, AMD_COMGR_DATA_KIND_SOURCE, 0, &dataOutput));
		CHECK_COMGR(amd_comgr_get_data(dataOutput, &size, NULL));
		
        char *bytes = (char *)malloc(size+1);
        bytes[size] = '\0';
		CHECK_COMGR(amd_comgr_get_data(dataOutput, &size, bytes));
        std::string strDisassembly(bytes);
        free(bytes);
        //CHECK_COMGR(amd_comgr_destroy_data_set(dataSetIn));
        CHECK_COMGR(amd_comgr_release_data(dataOutput));
        CHECK_COMGR(amd_comgr_release_data(executable));
        std::cout << strDisassembly << std::endl;
        parseDisassembly(strDisassembly);
    }
    return bReturn;
}

bool kernelDB::parseDisassembly(const std::string& text)
{
    bool bReturn = true;
    std::istringstream in(text);
    std::string line;
    parse_mode mode = BEGIN;
    std::string strKernel;
    while(std::getline(in,line))
    {
        auto it = line.begin();
        switch(mode)
        {
            case BEGIN:
                if (*it == ':')
                    mode = KERNEL;
                break;
            case KERNEL:
                it = --(line.end());
                if (*it == ':')
                {
                    if (line.find_first_of(" ") == std::string::npos)
                    {
                        strKernel = line.substr(0, line.length() - 1);
                        mode=BBLOCK;
                    }
                }
                break;
            case BBLOCK:
                break;
            case INSTRUCTION:
                break;
            default:
                break;
        }
    }
    return bReturn;
}

void kernelDB::getElfSectionBits(const std::string &fileName, const std::string &sectionName, std::vector<uint8_t>& sectionData ) {
    std::ifstream file(fileName, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file: " + fileName);
    }

    // Read ELF header
    Elf64_Ehdr elfHeader;
    file.read(reinterpret_cast<char*>(&elfHeader), sizeof(elfHeader));

    // Check if it's an ELF file
    if (memcmp(elfHeader.e_ident, ELFMAG, SELFMAG) != 0) {
        throw std::runtime_error("Not a valid ELF file");
    }

    // Seek to the section header table
    file.seekg(elfHeader.e_shoff, std::ios::beg);

    // Read all section headers
    std::vector<Elf64_Shdr> sectionHeaders(elfHeader.e_shnum);
    file.read(reinterpret_cast<char*>(sectionHeaders.data()), elfHeader.e_shnum * sizeof(Elf64_Shdr));

    // Seek to the section header string table
    const Elf64_Shdr &shstrtab = sectionHeaders[elfHeader.e_shstrndx];
    std::vector<char> shstrtabData(shstrtab.sh_size);
    file.seekg(shstrtab.sh_offset, std::ios::beg);
    file.read(shstrtabData.data(), shstrtab.sh_size);

    // Find the section by name
    for (const auto &section : sectionHeaders) {
        std::string currentSectionName(&shstrtabData[section.sh_name]);

        if (currentSectionName == sectionName) {
            // Read the section data
            sectionData.resize(section.sh_size);
            file.seekg(section.sh_offset, std::ios::beg);
            file.read(reinterpret_cast<char*>(sectionData.data()), section.sh_size);
            return;  // Return the section bits
        }
    }

    throw std::runtime_error("Section not found: " + sectionName);
}

}//kernelDB
