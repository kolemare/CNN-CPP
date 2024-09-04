/*
MIT License
Copyright (c) 2024 Marko Kostić

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

This project is the CNN-CPP Framework. Usage of this code is free, and 
uploading and using the code is also free, with a humble request to mention 
the origin of the implementation, the author Marko Kostić, and the repository 
link: https://github.com/kolemare/CNN-CPP.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
*/

#include "ThreadPool.hpp"

// Constructor to create a thread pool with given number of threads
ThreadPool::ThreadPool(size_t num_threads)
{
    // Creating worker threads
    for (size_t i = 0; i < num_threads; ++i)
    {
        threads_.emplace_back([this]
                              {
            while (true)
            {
                std::function<void()> task;
                {
                    // Locking the queue so that data can be shared safely
                    std::unique_lock<std::mutex> lock(queue_mutex_);

                    // Waiting until there is a task to execute or the pool is stopped
                    cv_.wait(lock, [this] {
                        return !tasks_.empty() || stop_;
                    });

                    // Exit the thread in case the pool is stopped and there are no tasks
                    if (stop_ && tasks_.empty())
                    {
                        return;
                    }

                    // Get the next task from the queue
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }

                task();
            } });
    }
}

// Destructor to stop the thread pool
ThreadPool::~ThreadPool()
{
    {
        // Lock the queue to update the stop flag safely
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }

    // Notify all threads
    cv_.notify_all();

    // Joining all worker threads to ensure they have completed their tasks
    for (auto &thread : threads_)
    {
        thread.join();
    }
}
