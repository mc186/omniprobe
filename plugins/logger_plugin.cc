#include "plugin.h"
#include "inc/message_logger.h"

extern "C"{
    PUBLIC_API void getMessageHandlers(const std::string& kernel, uint64_t dispatch_id, std::vector<dh_comms::message_handler_base *>& outHandlers)
    {
        std::string location = "console";
        const char* logDurLogLocation = std::getenv("LOGDUR_LOG_LOCATION");
        if (logDurLogLocation != NULL)
            location = logDurLogLocation;
        outHandlers.push_back(new message_logger_t(kernel, dispatch_id, location, false));
    }
}
