#ifndef CONFIG_RECEIVER_HPP
#define CONFIG_RECEIVER_HPP

#include <zmqpp/zmqpp.hpp>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/ecdsa.h>
#include <string>
#include <stdexcept>

using json = nlohmann::json;

class ConfigReceiver {
public:
    ConfigReceiver(const std::string& endpoint, const std::string& public_key_pem);
    ~ConfigReceiver();

    // Receive and process a configuration message
    json receive_config();

private:
    zmqpp::context_t context_;
    zmqpp::socket_t socket_;
    EVP_PKEY* public_key_;

    // Initialize OpenSSL public key from PEM string
    void init_public_key(const std::string& public_key_pem);

    // Verify ECDSA signature
    bool verify_signature(const json& config, const std::string& signature_b64);
};

#endif // CONFIG_RECEIVER_HPP
