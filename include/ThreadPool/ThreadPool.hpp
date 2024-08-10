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
