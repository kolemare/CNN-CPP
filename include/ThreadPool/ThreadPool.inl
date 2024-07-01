#ifndef THREADPOOL_INL_HPP
#define THREADPOOL_INL_HPP

#include "ThreadPool.hpp"

// Enqueue task for execution by the thread pool
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
