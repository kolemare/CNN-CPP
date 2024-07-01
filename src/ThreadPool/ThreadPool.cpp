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
