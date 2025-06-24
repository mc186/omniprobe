[
    {
        "name": "AddressLogger",
        "description": "Log raw memory traces",
        "lib_name": "libLogMessages64.so",
    },
    {
        "name": "BasicBlockLogger",
        "description": "Log raw timestamps from every basic block",
        "lib_name": "libLogMessages64.so",
        "llvm_plugin": "libAMDGCNSubmitBBStart-triton.so" 
    },
    {
        "name": "Heatmap",
        "description": "Produce a per dispatch memory heatmap",
        "lib_name": "libdefaultMessageHandlers64.so"
    },
    {
        "name": "MemoryAnalysis",
        "description": "Analyze memory access efficiency",
        "lib_name": "libMemAnalysis64.so"
    },
    {
        "name": "BasicBlockAnalysis",
        "description": "Analyze memory access efficiency",
        "lib_name": "libBasicBlocks64.so",
        "llvm_plugin": "libAMDGCNSubmitBBStart-triton.so" 
    }
]
