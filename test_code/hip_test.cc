
/******************************************************************************
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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
#include <fstream>
#include <vector>
#include <cstring>
#include <string>
#include <map>
#include <link.h>
#include <elf.h>
#include <amd_comgr.h>
#include <hsa.h>

#define CHECK_COMGR(call)                                                                          \
  if (amd_comgr_status_s status = call) {                                                          \
    const char* reason = "";                                                                       \
    amd_comgr_status_string(status, &reason);                                                      \
    std::cerr << __LINE__ << " code: " << status << std::endl;                                     \
    std::cerr << __LINE__ << " failed: " << reason << std::endl;                                   \
    exit(1);                                                                                       \
  }



int callback(struct dl_phdr_info *info, size_t size, void *data) {
    std::vector<std::string>* libraries = static_cast<std::vector<std::string>*>(data);
    
    if (info->dlpi_name && *info->dlpi_name) {  // Filter out empty names
        libraries->push_back(std::string(info->dlpi_name));
    }
    
    return 0;  // Continue iteration
}

std::vector<std::string> getSharedLibraries() {
    std::vector<std::string> libraries;
    dl_iterate_phdr(callback, &libraries);
    return libraries;
}



template<typename T>
struct hsa_cmp
{
    bool operator() (const T& first, const T& second) const
    {
        return first.handle < second.handle;
    }
};
amd_comgr_status_t iterateMetadata(amd_comgr_metadata_node_t key, amd_comgr_metadata_node_t value, void *userData);

void dumpList(amd_comgr_metadata_node_t value, std::string *pIndent)
{
    size_t listSize;
    char szBuff[1024];
    size_t size;
    amd_comgr_metadata_kind_t valueKind;
    amd_comgr_get_metadata_list_size(value, &listSize);
    printf("%sNumber of list elements: %lu\n", pIndent->c_str(), listSize);
    for (unsigned int i = 0; i < listSize; i++)
    {
        amd_comgr_metadata_node_t thisValue;
        amd_comgr_index_list_metadata(value, i, &thisValue);
        amd_comgr_get_metadata_kind(thisValue, &valueKind);
        //printf("%sList index %u Value Kind: %d\n", pIndent->c_str(), i, valueKind);
        switch(valueKind)
        {
            case AMD_COMGR_METADATA_KIND_LIST:
                {
                    std::string strIndent(*pIndent);
                    strIndent += "\t";
                    printf("%s\tIndex %u is a LIST\n", pIndent->c_str(), i);
                    dumpList(thisValue, &strIndent);
                }
                break;
            case AMD_COMGR_METADATA_KIND_STRING:
                {
                    size = sizeof(szBuff);
                    amd_comgr_get_metadata_string(thisValue, &size, szBuff);
                    printf("%s\tValue:%s\n", pIndent->c_str(), szBuff);
                }
                break;
            case AMD_COMGR_METADATA_KIND_MAP:
                {
                    std::string strIndent(*pIndent);
                    printf("%s\tIndex %u is a MAP\n", pIndent->c_str(), i);
                    strIndent += "\t\t";
                    amd_comgr_iterate_map_metadata(thisValue,iterateMetadata,static_cast<void *>(&strIndent));
                }
                break;
            default:
                break;
        }
    }
}

amd_comgr_status_t iterateMetadata(amd_comgr_metadata_node_t key, amd_comgr_metadata_node_t value, void *userData)
{
        char szBuff[1024];
        size_t size;
        std::string *pIndent = static_cast<std::string *>(userData);
        amd_comgr_metadata_kind_t keyKind, valueKind;
        amd_comgr_get_metadata_kind(key, &keyKind);
        amd_comgr_get_metadata_kind(value, &valueKind);
        memset(szBuff,0,sizeof(szBuff));

        if (keyKind == AMD_COMGR_METADATA_KIND_STRING)
        {
            size = sizeof(szBuff);
            amd_comgr_get_metadata_string(key, &size, szBuff);
            printf("%sKey:%s\n", pIndent->c_str(), szBuff);
        }

        if (valueKind == AMD_COMGR_METADATA_KIND_LIST)
        {
            size_t listSize;
            amd_comgr_get_metadata_list_size(value, &listSize);
            printf("%sNumber of list elements: %lu\n", pIndent->c_str(), listSize);
            for (unsigned int i = 0; i < listSize; i++)
            {
                amd_comgr_metadata_node_t thisValue;
                amd_comgr_index_list_metadata(value, i, &thisValue);
                amd_comgr_get_metadata_kind(thisValue, &valueKind);
                //printf("%sList index %u Value Kind: %d\n", pIndent->c_str(), i, valueKind);
                switch(valueKind)
                {
                    case AMD_COMGR_METADATA_KIND_LIST:
                        {
                            std::string strIndent(*pIndent);
                            strIndent += "\t";
                            printf("%s\tList Index %u is a LIST\n", pIndent->c_str(), i);
                            dumpList(thisValue, &strIndent);
                        }
                        break;
                    case AMD_COMGR_METADATA_KIND_STRING:
                        {
                            size = sizeof(szBuff);
                            amd_comgr_get_metadata_string(thisValue, &size, szBuff);
                            printf("%s\tList index %u is a string of Value:%s\n", pIndent->c_str(), i, szBuff);
                        }
                        break;
                    case AMD_COMGR_METADATA_KIND_MAP:
                        {
                            std::string strIndent(*pIndent);
                            printf("%s\tList Index %u is a MAP\n", pIndent->c_str(), i);
                            strIndent += "\t\t";
                            amd_comgr_iterate_map_metadata(thisValue,iterateMetadata,static_cast<void *>(&strIndent));
                        }
                        break;
                    default:
                        break;
                }
            }
        }
        else if (valueKind == AMD_COMGR_METADATA_KIND_STRING)
        {
            size = sizeof(szBuff);
            amd_comgr_get_metadata_string(value, &size, szBuff);
            printf("%sValue:%s\n", pIndent->c_str(), szBuff);
        }
        else if (valueKind == AMD_COMGR_METADATA_KIND_MAP)
        {
            std::string strIndent(*pIndent);
            strIndent += "\t";
            amd_comgr_iterate_map_metadata(value, iterateMetadata, static_cast<void *>(&strIndent));
        }
        else
            printf("COULDN'T PROCESS NODE OF TYPE %d\n", valueKind);
        return AMD_COMGR_STATUS_SUCCESS;
}



amd_comgr_status_t printSymbol(amd_comgr_symbol_t symbol, void *userData) {
  amd_comgr_status_t status;

  size_t nlen;
  CHECK_COMGR(amd_comgr_symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_NAME_LENGTH,
                                     (void *)&nlen));

  char *name = (char *)malloc(nlen + 1);
  CHECK_COMGR(amd_comgr_symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_NAME,
                                     (void *)name));

  amd_comgr_symbol_type_t type;
  CHECK_COMGR(amd_comgr_symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_TYPE,
                                     (void *)&type));

  uint64_t size;
  CHECK_COMGR(amd_comgr_symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_SIZE,
                                     (void *)&size));

  bool undefined;
  CHECK_COMGR(amd_comgr_symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_IS_UNDEFINED,
                                     (void *)&undefined));

  uint64_t value;
  CHECK_COMGR(amd_comgr_symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_VALUE,
                                     (void *)&value));

  printf("%d:  name=%s, type=%d, size=%lu, undef:%d, value:%lu\n",
         *(int *)userData, name, type, size, undefined ? 1 : 0, value);
  *(int *)userData += 1;


  free(name);

  return status;
}
// Function to read the bits of a specific section from an ELF binary
std::vector<uint8_t> getElfSectionBits(const std::string &fileName, const std::string &sectionName) {
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
            std::vector<uint8_t> sectionData(section.sh_size);
            file.seekg(section.sh_offset, std::ios::beg);
            file.read(reinterpret_cast<char*>(sectionData.data()), section.sh_size);
            return sectionData;  // Return the section bits
        }
    }

    throw std::runtime_error("Section not found: " + sectionName);
}


bool write_bits(std::vector<uint8_t>& bits, std::string& strFileName)
{
    std::ofstream file(strFileName, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file: " + strFileName);
    }

    file.write(reinterpret_cast<const char *>(bits.data()), bits.size());
    file.close();
    return true;
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
                pName[length] = '\0';
                status = hsa_isa_get_info(isa, HSA_ISA_INFO_NAME, 0, pName);
                std::cerr << "Isa name: " << pName << std::endl;
                if (status == HSA_STATUS_SUCCESS)
                    pList->push_back(std::string(pName));
           }
           return HSA_STATUS_SUCCESS;
        }, reinterpret_cast<void *>(&list));   
    return list;
}

std::map<hsa_agent_t, std::vector<std::string>, hsa_cmp<hsa_agent_t>> getIsaByAgent()
{
	std::map<hsa_agent_t, std::vector<std::string>, hsa_cmp<hsa_agent_t>> agents;

    hsa_status_t status = hsa_iterate_agents ([](hsa_agent_t agent, void *data){
                    std::map<hsa_agent_t, std::vector<std::string>, hsa_cmp<hsa_agent_t>> *isas  = reinterpret_cast<std::map<hsa_agent_t, std::vector<std::string>, hsa_cmp<hsa_agent_t>> *>(data);
					hsa_device_type_t type;
                    hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, static_cast<void *>(&type));
					if (status == HSA_STATUS_SUCCESS && type == HSA_DEVICE_TYPE_GPU)
                    {
                        std::cerr << "Found a gpu" << std::endl;
                        
						(*isas)[agent] = getIsaList(agent);
                    }
                    return HSA_STATUS_SUCCESS;
                }, reinterpret_cast<void *>(&agents));
    std::cerr << "HSA_STATUS = " << std::hex << status << std::endl;
    return agents;
}

int main() {
    try {
        hsa_init();
        //std::string fileName = "/work1/amd/klowery/dh_comms/build/examples/bin/heatmap_example";
        std::string fileName = "/work1/amd/klowery/logduration/src/test/quicktest";
        std::string sectionName = ".hip_fatbin"; // Example section
        std::vector<uint8_t> sectionBits = getElfSectionBits(fileName, sectionName);

        std::cout << "Section " << sectionName << " has " << sectionBits.size() << " bytes\n";
        std::string filename = "./foo.hsaco";
        amd_comgr_data_t executable, bundle;
        CHECK_COMGR(amd_comgr_create_data(AMD_COMGR_DATA_KIND_FATBIN, &bundle));
        CHECK_COMGR(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &executable));
        CHECK_COMGR(amd_comgr_set_data(bundle, sectionBits.size(), reinterpret_cast<const char *>(sectionBits.data())));
        std::cerr << "WOO HOO!\n";
        std::map<hsa_agent_t, std::vector<std::string>, hsa_cmp<hsa_agent_t>> agents = getIsaByAgent();
        if (agents.size())
        {
            auto it = agents.begin();
            for(auto isa : it->second)
                std::cerr << isa << std::endl;
            amd_comgr_code_object_info_t querylist = {it->second[0].c_str(), 0, 0};
            CHECK_COMGR(amd_comgr_lookup_code_object(bundle,&querylist, 1));
            CHECK_COMGR(amd_comgr_set_data(executable, querylist.size, reinterpret_cast<const char *>(sectionBits.data() + querylist.offset)));
            std::cerr << "GOT AND EXECUTABLE FROM A BINARY!!!!" << std::endl;
        }
        amd_comgr_metadata_node_t metadata;

        /*CHECK_COMGR(amd_comgr_get_data_metadata(executable, &metadata));
    
        amd_comgr_metadata_kind_t md_kind;
        CHECK_COMGR(amd_comgr_get_metadata_kind(metadata, &md_kind));
        if (md_kind == AMD_COMGR_METADATA_KIND_MAP)
        {
            std::string strIndent("");
            //std::map<std::string, arg_descriptor_t> parms; 
            //computeKernargData(metadata, parms);
            CHECK_COMGR(amd_comgr_iterate_map_metadata(metadata,iterateMetadata,static_cast<void *>(&strIndent))); 
        }*/


        write_bits(sectionBits, filename);


        // Further processing can be done here

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
    }

    return 0;
}

