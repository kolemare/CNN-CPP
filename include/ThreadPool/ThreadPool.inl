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

#ifndef THREADPOOL_INL_HPP
#define THREADPOOL_INL_HPP

#include "ThreadPool.hpp"

/**
 * @brief Enqueue a task for execution by the thread pool.
 *
 * This method adds a new task to the task queue for execution by the thread pool's worker threads.
 * The task is wrapped in a `std::packaged_task` to allow the result to be retrieved asynchronously.
 *
 * @tparam F The type of the function to execute.
 * @tparam Args The types of the arguments to pass to the function.
 * @param f The function to execute.
 * @param args The arguments to pass to the function.
 * @return std::future<typename std::result_of<F(Args...)>::type> A future to retrieve the result of the task.
 */
template <class F, class... Args>
auto ThreadPool::enqueue(F &&f, Args &&...args) -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        tasks_.emplace([task]()
                       { (*task)(); });
    }
    cv_.notify_one();
    return res;
}

#endif // THREADPOOL_INL_HPP
