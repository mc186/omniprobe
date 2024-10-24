#include "plugin.h"

extern "C"{
    PUBLIC_API void getMessageHandlers(const std::string& kernel, uint64_t dispatch_id, std::vector<std::unique_ptr<dh_comms::message_handler_base>&&>& outHandlers)
    {
    }
}
