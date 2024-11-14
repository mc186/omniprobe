#include "plugin.h"
#include "inc/time_interval_handler_wrapper.h"
#include "inc/memory_heatmap_wrapper.h"

extern "C"{
    PUBLIC_API void getMessageHandlers(const std::string& kernel, uint64_t dispatch_id, std::vector<dh_comms::message_handler_base *>& outHandlers)
    {
        outHandlers.push_back(new time_interval_handler_wrapper(kernel, dispatch_id,false));
        outHandlers.push_back(new memory_heatmap_wrapper(kernel, dispatch_id, 1024*1024, false));
    }
}
