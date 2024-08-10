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
