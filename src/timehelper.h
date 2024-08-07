#pragma once

#include <chrono>

class timeHelper
{
public:
    timeHelper()
    {
        clock_gettime(CLOCK_MONOTONIC, &ts_start);
    }
    uint64_t getStartTime()
    {
        return (ts_start.tv_sec * 1000000000) + ts_start.tv_nsec;
        //return std::chrono::time_point_cast<std::chrono::nanoseconds>(start).time_since_epoch().count();
    }
    void reset()
    {
        clock_gettime(CLOCK_MONOTONIC, &ts_start);
        start = std::chrono::steady_clock::now();
    }
    uint64_t getElapsedNanos()
    {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds> (end - start).count();
    }
    uint64_t getElapsedMicros()
    {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::nanoseconds> (end - start).count()) / 1000;
    }
private:
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    struct timespec ts_start;
};
