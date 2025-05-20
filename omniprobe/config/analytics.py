[
    {
        "name": "MessageLogger",
        "description": "Log raw instrumentation messages",
        "lib_name": "libLogMessages64.so",
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
        "llvm_plugin": "libAMDGCNSubmitBBInterval-triton.so"
        #"llvm_plugin": "libAMDGCNSubmitBBStart-triton.so" 
    }
]
