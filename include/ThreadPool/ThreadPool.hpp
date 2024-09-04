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

#ifndef THREADPOOL_HPP
#define THREADPOOL_HPP

#include "Common.hpp"

/**
 * @class ThreadPool
 * @brief A thread pool for executing tasks concurrently.
 */
class ThreadPool
{
public:
    /**
     * @brief Constructor to create a thread pool with a given number of threads.
     * @param num_threads The number of threads in the pool. Defaults to the number of hardware concurrency.
     */
    ThreadPool(size_t num_threads = std::thread::hardware_concurrency());

    /**
     * @brief Destructor to stop the thread pool.
     */
    ~ThreadPool();

    /**
     * @brief Enqueue a task for execution by the thread pool.
     * @tparam F The function type.
     * @tparam Args The argument types.
     * @param f The function to execute.
     * @param args The arguments to pass to the function.
     * @return A std::future that can be used to get the result of the task.
     */
    template <class F, class... Args>
    auto enqueue(F &&f, Args &&...args) -> std::future<typename std::result_of<F(Args...)>::type>;

private:
    std::vector<std::thread> threads_;        /**< Vector to store worker threads */
    std::queue<std::function<void()>> tasks_; /**< Queue of tasks */
    std::mutex queue_mutex_;                  /**< Mutex to synchronize access to shared data */
    std::condition_variable cv_;              /**< Condition variable to signal changes in the state of the tasks queue */
    bool stop_ = false;                       /**< Flag to indicate whether the thread pool should stop or not */
};

#include "ThreadPool.inl"

#endif // THREADPOOL_HPP
