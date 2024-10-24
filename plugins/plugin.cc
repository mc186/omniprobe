#include "plugin.h"

extern "C"{
    PUBLIC_API void getMessageHandlers(std::vector<std::unique_ptr<dh_comms::message_handler_base>&&>& outHandlers)
    {
    }
}
