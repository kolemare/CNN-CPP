#ifndef THREADPOOL_HPP
#define THREADPOOL_HPP

#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <future>

class ThreadPool
{
public:
    // Constructor to create a thread pool with a given number of threads
    ThreadPool(size_t num_threads = std::thread::hardware_concurrency());

    // Destructor to stop the thread pool
    ~ThreadPool();

    // Enqueue task for execution by the thread pool
    template <class F, class... Args>
    auto enqueue(F &&f, Args &&...args) -> std::future<typename std::result_of<F(Args...)>::type>;

private:
    // Vector to store worker threads
    std::vector<std::thread> threads_;

    // Queue of tasks
    std::queue<std::function<void()>> tasks_;

    // Mutex to synchronize access to shared data
    std::mutex queue_mutex_;

    // Condition variable to signal changes in the state of the tasks queue
    std::condition_variable cv_;

    // Flag to indicate whether the thread pool should stop or not
    bool stop_ = false;
};

#include "ThreadPool.inl"

#endif // THREADPOOL_HPP
