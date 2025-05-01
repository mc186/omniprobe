#include <vector>
#include <string>
#include <filesystem>
#include <ctime>

std::vector<std::string> getRecentFiles(const std::string& dirName, const std::time_t& timeThreshold) {
    std::vector<std::string> recentFiles;
    namespace fs = std::filesystem;
    
    try {
        for (const auto& entry : fs::directory_iterator(dirName)) {
            if (fs::is_regular_file(entry)) {
                auto lastWriteTime = fs::last_write_time(entry);
                auto timeSinceEpoch = lastWriteTime.time_since_epoch();
                auto seconds = std::chrono::duration_cast<std::chrono::seconds>(timeSinceEpoch).count();
                
                if (seconds > timeThreshold) {
                    recentFiles.push_back(entry.path().filename().string());
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        // Handle directory access errors silently
        return recentFiles;
    }
    
    return recentFiles;
}
