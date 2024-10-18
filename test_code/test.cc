#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <map>
#include <amd_comgr.h>

#define CHECK_COMGR(call)                                                                          \
  if (amd_comgr_status_s status = call) {                                                          \
    const char* reason = "";                                                                       \
    amd_comgr_status_string(status, &reason);                                                      \
    std::cerr << __LINE__ << " code: " << status << std::endl;                                     \
    std::cerr << __LINE__ << " failed: " << reason << std::endl;                                   \
    exit(1);                                                                                       \
  }


amd_comgr_status_t iterateMetadata(amd_comgr_metadata_node_t key, amd_comgr_metadata_node_t value, void *userData);

typedef struct arg_descriptor {
    size_t explicit_args_length;
    size_t hidden_args_length;
    size_t kernarg_length;
}arg_descriptor_t;

std::string get_metadata_string(amd_comgr_metadata_node_t node)
{
    std::string strValue;
    amd_comgr_metadata_kind_t kind;
    CHECK_COMGR(amd_comgr_get_metadata_kind(node, &kind));
    if (kind == AMD_COMGR_METADATA_KIND_STRING)
    {
        size_t size;
        CHECK_COMGR(amd_comgr_get_metadata_string(node, &size, NULL));
        strValue.resize(size);
        CHECK_COMGR(amd_comgr_get_metadata_string(node, &size, &strValue[0]));
    }
    return strValue;
}

void computeKernargData(amd_comgr_metadata_node_t exec_map, std::map<std::string, arg_descriptor>& arg_data)
{
    amd_comgr_metadata_node_t kernels;
    amd_comgr_metadata_node_t args;
    amd_comgr_metadata_node_t kernarg_length;
    CHECK_COMGR(amd_comgr_metadata_lookup(exec_map, "amdhsa.kernels", &kernels));
    size_t size;
    CHECK_COMGR(amd_comgr_get_metadata_list_size(kernels, &size));
    for (size_t i = 0; i < size; i++)
    {
        amd_comgr_metadata_node_t value;
        amd_comgr_metadata_kind_t kind;
        CHECK_COMGR(amd_comgr_index_list_metadata(kernels, i, &value));
        CHECK_COMGR(amd_comgr_get_metadata_kind(value, &kind));

        if (kind == AMD_COMGR_METADATA_KIND_MAP)
        {
            amd_comgr_metadata_node_t field;
            CHECK_COMGR(amd_comgr_metadata_lookup(value,".symbol", &field));
            std::string strName = get_metadata_string(field);
            arg_descriptor_t desc = {0,0,0};
            amd_comgr_metadata_node_t args;
            CHECK_COMGR(amd_comgr_metadata_lookup(value, ".args", &args));
            CHECK_COMGR(amd_comgr_get_metadata_kind(args, &kind));
            amd_comgr_metadata_node_t kernarg_length;
            CHECK_COMGR(amd_comgr_metadata_lookup(value, ".kernarg_segment_size", &kernarg_length));
            desc.kernarg_length = std::stoul(get_metadata_string(kernarg_length));
            if (kind == AMD_COMGR_METADATA_KIND_LIST)
            {
                size_t arg_count;
                CHECK_COMGR(amd_comgr_get_metadata_list_size(args, &arg_count));
                std::cerr << "Kernel has a list of size " << arg_count << std::endl;
                for (size_t j = 0; j < arg_count; j++)
                {
                    amd_comgr_metadata_node_t parm_map;
                    amd_comgr_metadata_kind_t parm_kind;
                    CHECK_COMGR(amd_comgr_index_list_metadata(args, j, &parm_map));
                    CHECK_COMGR(amd_comgr_get_metadata_kind(parm_map,&parm_kind));
                    if (parm_kind == AMD_COMGR_METADATA_KIND_MAP)
                    {
                        /*
                         * List Index 3 is a MAP
                                Key:.offset
                                Value:24
                                Key:.size
                                Value:4
                                Key:.value_kind
                                Value:by_value
                            List Index 4 is a MAP
                                Key:.offset
                                Value:32
                                Key:.size
                                Value:8
                                Key:.value_kind
                                Value:hidden_global_offset_x

                         */
                        amd_comgr_metadata_node_t parm_size, parm_type;
                        CHECK_COMGR(amd_comgr_metadata_lookup(parm_map, ".size", &parm_size));
                        CHECK_COMGR(amd_comgr_metadata_lookup(parm_map, ".value_kind", &parm_type));
                        size_t arg_size = std::stoul(get_metadata_string(parm_size));
                        std::string parm_name = get_metadata_string(parm_type);
                        std::cerr << "parm_name = " << parm_name << std::endl;
                        if (parm_name.rfind("hidden_",0) == 0)
                            desc.hidden_args_length += arg_size;
                        else
                            desc.explicit_args_length += arg_size;
                    }
                }
            }
            std::cerr << strName << ": {" << desc.explicit_args_length << ", " << desc.hidden_args_length << ", " << desc.kernarg_length << "}\n";
        }
    }
}

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


int main(int argc, char **argv)
{
    std::string file_name("/home1/klowery/.triton/cache/96f1014ec3853c050dbbf89a464d119176d035306e5cdc28cea6b651bd395349/add_kernel.hsaco");
    amd_comgr_data_t executable;
    std::vector<char> buff;
    std::ifstream file(file_name, std::ios::binary | std::ios::ate);

    if (!file.is_open()) {
        std::cerr << "Failed to open the file: " << file_name << std::endl;
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
    std::cerr << "SUCCESS CREATING EXECUTABLE\n";
    int symbolCount;
    amd_comgr_iterate_symbols(executable, printSymbol, static_cast<void *>(&symbolCount));
    amd_comgr_metadata_node_t metadata;

    CHECK_COMGR(amd_comgr_get_data_metadata(executable, &metadata));

    amd_comgr_metadata_kind_t md_kind;
    CHECK_COMGR(amd_comgr_get_metadata_kind(metadata, &md_kind));

    if (md_kind == AMD_COMGR_METADATA_KIND_MAP)
    {
        std::string strIndent("");
        std::map<std::string, arg_descriptor_t> parms; 
        computeKernargData(metadata, parms);
        //CHECK_COMGR(amd_comgr_iterate_map_metadata(metadata,iterateMetadata,static_cast<void *>(&strIndent))); 
    }
}
