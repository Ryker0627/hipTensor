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

#include "include/timer.hpp"
#include <string>
#include <utility>
#include <iostream>

namespace hiptensor
{
    void Timer::start(std::string sItemName) 
    {
        auto curTime = std::chrono::high_resolution_clock::now();

        int idx = findItem(sItemName);
        if(idx==-1)
        {
            // if time counting item not exist, add new item name and set current time
            mTimeNames.push_back(sItemName);
            mTimeStats.push_back(timeStat{curTime,curTime,0});
        }
        else
        {
            // if time counting item exists, reset time to current time
            mTimeStats[idx].start = curTime;
            mTimeStats[idx].end = curTime;
            mTimeStats[idx].duration = 0;
        }
    }

    void Timer::end(std::string sItemName) 
    {
        int idx = findItem(sItemName);
        if(idx<0) return;

        auto curTime = std::chrono::high_resolution_clock::now();
        // Based on the countint item end time, calculate the duration time in ms
        mTimeStats[idx].end = curTime;
        //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(mTimeStats[idx].end-mTimeStats[idx].start);
        //mTimeStats[idx].duration = duration.count();
        std::chrono::duration<double, std::milli> fp_ms = mTimeStats[idx].end-mTimeStats[idx].start;
        mTimeStats[idx].duration = fp_ms.count();
    }


    void Timer::set_duration(std::string sItemName, double duration) 
    {
        int idx = findItem(sItemName);
        if(idx<0) return;
 
        auto curTime = std::chrono::high_resolution_clock::now();
        mTimeStats[idx].end = curTime;
        mTimeStats[idx].duration = duration;
        mTimeStats[idx].start = curTime;
    }

    int Timer::findItem(std::string sItemName)
    {
        int idx=-1;
        for(int i=0;i<mTimeNames.size();i++) {
            if(mTimeNames[i]==sItemName) {
                idx=i;
                break;
            }
        }
        return idx;
    }

    void Timer::report()
    {
        //report time items
        int size = mTimeNames.size();
        for(int idx = 0; idx < size; ++idx) {
            std::cout << mTimeNames[idx] << ": " << mTimeStats[idx].duration << "  ms" << std::endl;
        }
    }
}