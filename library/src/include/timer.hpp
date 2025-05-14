/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

 #ifndef HIPTENSOR_TIMER_STAT_HPP
 #define HIPTENSOR_TIMER_STAT_HPP

 #include <vector>
 #include <chrono>
 #include "singleton.hpp"

namespace hiptensor 
{
    class Timer : public LazySingleton<Timer> 
    {
    private:
        using time_point = std::chrono::high_resolution_clock::time_point;
        struct timeStat {
            time_point start;
            time_point end;
            double duration;
        };

        // item names for time counting
        std::vector<std::string> mTimeNames;
        // first item of the pair -- start time for the item (ms)
        // second item of the pair -- end time for the item (ms)
        std::vector<timeStat> mTimeStats;
    public:
        void start(std::string sItemName);
        void end(std::string sItemName);  
        void set_duration(std::string sItemName, double duration);
        void report();
    private:
        // find a time counting item
        int findItem(std::string sItemName);
    };
}  // namespace hiptensor


 #endif // HIPTENSOR_TIMER_STAT_HPP