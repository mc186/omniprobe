#include <iostream>
#include <cassert>
#include <set>
#include "inc/basic_block_analysis.h"
#include "time_interval_handler.h"
#include <iomanip>
#include <fstream>
#include <cstdint>
#include <stdexcept>

std::vector<std::string> readFileLines(const std::string& filename, uint32_t startLine, uint32_t endLine) {
    std::vector<std::string> lines;

    // Validate input parameters
    if (startLine == 0 || startLine > endLine) {
        throw std::invalid_argument("Invalid line range: startLine must be >= 1 and <= endLine");
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    std::string line;
    uint32_t currentLine = 0;

    // Read until we reach startLine or EOF
    while (currentLine < startLine - 1 && std::getline(file, line)) {
        ++currentLine;
    }

    // Read lines from startLine to endLine inclusive
    while (currentLine < endLine && std::getline(file, line)) {
        lines.push_back(line);
        ++currentLine;
    }

    file.close();
    return lines;
}

void printVectorsSideBySide(const std::vector<std::string>& vec1, const std::vector<std::string>& vec2) {
    // Find the longest string in each vector for column width
    size_t maxLen1 = 0, maxLen2 = 0;
    for (const auto& str : vec1) {
        maxLen1 = std::max(maxLen1, str.length());
    }
    for (const auto& str : vec2) {
        maxLen2 = std::max(maxLen2, str.length());
    }

    // Print vectors side by side
    size_t maxRows = std::max(vec1.size(), vec2.size());
    for (size_t i = 0; i < maxRows; ++i) {
        // Print first vector element (or empty if index exceeds size)
        std::cout << std::left << std::setw(maxLen1 + 2);
        if (i < vec1.size()) {
            std::cout << vec1[i];
        } else {
            std::cout << "";
        }

        // Print second vector element (or empty if index exceeds size)
        std::cout << std::left << std::setw(maxLen2 + 2);
        if (i < vec2.size()) {
            std::cout << vec2[i];
        } else {
            std::cout << "";
        }

        std::cout << std::endl;
    }
}

uint32_t countSetBits(uint64_t mask) {
    uint32_t count = 0;
    while (mask) {
        count += mask & 1;
        mask >>= 1;
    }
    return count;
}

basic_block_analysis::basic_block_analysis(const std::string& strKernel, uint64_t dispatch_id, std::string& strLocation, bool verbose)
    : first_start_(0xffffffffffffffff),
      last_stop_(0),
      total_time_(0),
      no_intervals_(0),
      verbose_(verbose),
      strKernel_(strKernel),
      dispatch_id_(dispatch_id),
      current_block_(nullptr),
      start_time_(0)
{
}

basic_block_analysis::~basic_block_analysis()
{
}

bool basic_block_analysis::handle(const dh_comms::message_t &message, const std::string& kernel, kernelDB::kernelDB& kdb)
{
    bool bReturn = true;
    auto hdr = message.wave_header();
    waveIdentifier_t wave = {hdr.block_idx_x, hdr.block_idx_y, hdr.block_idx_z, hdr.wave_num};

    try
    {
        uint32_t block_idx = hdr.user_data;
        auto& thisKernel = kdb.getKernel(kernel);
        const auto& blocks = thisKernel.getBasicBlocks();
        assert(block_idx < blocks.size());
        kernelDB::basicBlock *thisBlock = blocks[block_idx].get();
        auto& instructions = thisBlock->getInstructions();
        if (instructions.size())
        {
            std::set<kernelDB::basicBlock *> blocks;
            for (auto& in : instructions)
                blocks_seen_.insert(in.block_);
            auto biit = block_info_.find(thisBlock);
    //        std::cout << "Message Type: " << hdr.user_type << std::endl;
            if (hdr.user_type == dh_comms::message_type::time_interval)
            {
                dh_comms::time_interval ti = *(const dh_comms::time_interval *)message.data_item(0);
                if (biit != block_info_.end())
                {
                    biit->second.count_++;
                    biit->second.duration_ += ti.stop - ti.start;
                }
                else
                {
                    //assert(ti.stop - ti.start != 0);
                    block_info_[thisBlock] = {1, ti.stop - ti.start};
                }
            }
            else
            {
                auto wsit = wave_states_.find(wave);
                // If we have a state cached for this wave, use that data to update the block
                // count and duration
                // Otherwise, just set the wave state
                if (wsit != wave_states_.end())
                {
                    // If we have seen this block already
                    // increment the count and add the current duration 
                    // otherwise, just set the count to 1 and set the duration.
                    biit = block_info_.find(wsit->second.current_block_);
                    if (biit != block_info_.end())
                    {
                        biit->second.count_+=1;
                        biit->second.duration_ += hdr.timestamp - wsit->second.start_time_;
                    }
                    else
                    {
                        block_info_[wsit->second.current_block_] = {1, hdr.timestamp - wsit->second.start_time_};
                    }
                    
                    // Need to check for s_endpgm and clean up wave state here
                    if (instructions[instructions.size() - 1].inst_ == "s_endpgm")
                    {
                        wave_states_.erase(wsit);
                    }
                    else
                    {
                        wsit->second.current_block_ = thisBlock;
                        wsit->second.start_time_ = hdr.timestamp;
                    }
                }
                else
                {
                    wave_states_[wave] = {thisBlock, hdr.timestamp, 1};
                }
            }
        }
    }
    catch (std::runtime_error e)
    {
        std::cerr << e.what() << std::endl;
        return false;
    }
    return bReturn;
}

bool basic_block_analysis::handle(const dh_comms::message_t &message)
{
    return true;
}

void basic_block_analysis::report(const std::string& kernel_name, kernelDB::kernelDB& kdb)
{
    std::cerr << "omniprobe basic block analysis for kernel " << strKernel_ << "[" << dispatch_id_ << "]\n";
    auto it = block_info_.begin();
    while (it != block_info_.end())
    {
        std::vector<std::string> isa, files;
        auto instructions = it->first->getInstructions();
        for (auto inst : instructions)
        {
            isa.push_back(inst.disassembly_);
            files.push_back(kdb.getFileName(kernel_name, inst.path_id_));
        }
        std::cerr << instructions[0].line_ << "," << instructions[instructions.size() - 1].line_ << "," << it->second.duration_ << "," << kdb.getFileName(kernel_name, instructions[0].path_id_) 
            << "," << it->second.count_ << std::endl;

        printVectorsSideBySide(isa, files);

        
        it++;
    }
}

void basic_block_analysis::report()
{
    std::cerr << "omniprobe time_interval report for kernel " << strKernel_ << " dispatch[" << std::dec << dispatch_id_ << "]\n";

}

void basic_block_analysis::clear()
{
    strKernel_ = "";
    dispatch_id_ = 0;
}
