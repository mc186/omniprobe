
#include "dh_comms.h"
#include "message_handlers.h"
#define PUBLIC_API __attribute__((visibility("default")))
#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API __attribute__((destructor))
extern "C"{

    PUBLIC_API void getMessageHandlers(const std::string& kernel, uint64_t dispatch_id, std::vector<dh_comms::message_handler_base*>& outHandlers);

}

typedef void (*getMessageHandlers_t)(const std::string& kernel, uint64_t dispatch_id, std::vector<dh_comms::message_handler_base *>& outHandlers);
