#pragma once

#include <Python.h>
#include <iostream>
#include <mutex>
#include <stdexcept>

class pythonMessageHandler {
public:
    pythonMessageHandler(std::string& module_name);
    ~pythonMessageHandler();
    void addressMessage(void *address);
    void timingMessage(uint64_t time);
private:
    std::string module_name_;
    PyObject *pModule_;
    PyObject *pAddressFunc_;
    PyObject *pTimingFunc_;
    static std::mutex global_mutex_;
};

