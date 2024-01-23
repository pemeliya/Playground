// Author: Kirk Saunders (ks825016@ohio.edu)
// Description: Simple implementation of a thread barrier
//              using C++ condition variables.
// Date: 2/17/2020

#ifndef THREADING_HPP
#define THREADING_HPP

#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <functional>
#include "common.h"

class Barrier {
 public:
    // Construct barrier for use with num threads.
    explicit Barrier(std::size_t num)
        : num_threads(num), wait_count(0), instance(0)
    {  }

    // disable copying of barrier
    Barrier(const Barrier&) = delete;
    Barrier& operator =(const Barrier&) = delete;

    // This function blocks the calling thread until
    // all threads (specified by num_threads) have
    // called it. Blocking is achieved using a
    // call to condition_variable.wait().
    void wait() {
        std::unique_lock<std::mutex> lock(mut); // acquire lock
        std::size_t inst = instance; // store current instance for comparison
                                     // in predicate

        if (++wait_count == num_threads) { // all threads reached barrier
            wait_count = 0; // reset wait_count
            instance++; // increment instance for next use of barrier and to
                        // pass condition variable predicate
            cv.notify_all();
        } else { // not all threads have reached barrier
            cv.wait(lock, [this, &inst]() { return instance != inst; });
            // NOTE: The predicate lambda here protects against spurious
            //       wakeups of the thread. As long as this->instance is
            //       equal to inst, the thread will not wake.
            //       this->instance will only increment when all threads
            //       have reached the barrier and are ready to be unblocked.
        }
    }
 private:
    std::size_t num_threads; // number of threads using barrier
    std::size_t wait_count; // counter to keep track of waiting threads
    std::size_t instance; // counter to keep track of barrier use count
    std::mutex mut; // mutex used to protect resources
    std::condition_variable cv; // condition variable used to block threads
};


struct ThreadPool {

  using JobFunc = std::function<void(int)>;//llvm::function_ref<void(int)>;
  explicit ThreadPool(size_t nThreads) : m_threads(nThreads) {

    for(size_t i = 0; i < nThreads; i++) {
      m_threads[i] = std::thread{&ThreadPool::threadFunc, this, i};
    }
  }

  void runJob(JobFunc f) {
    m_func = f;
    m_currentJobID += m_threads.size();
    m_jobCv.notify_all();
    wait();
  }

  void wait() {
    std::unique_lock lock(m_finishedMtx); 
    m_finishedCv.wait(lock, [this](){
      return !m_isRunning || m_arrived == m_currentJobID;
    });
  }

  ~ThreadPool() try {
    m_isRunning = false;
    m_jobCv.notify_all();
    for(auto& th: m_threads) {
      if(th.joinable())
        th.join();
    }
    VLOG("Thread pool joined");
  }
  catch(...) {
    PRINTZ("Exception when destroying thread pool!");
  }

private:

  void threadFunc(int id) try {
    m_isRunning = true;
    uint32_t localJobId = 0;
    while(m_isRunning) 
    {
      // VLOG("My job ID: " << localJobId);
      {
        std::unique_lock lock(m_jobMtx); 
        m_jobCv.wait(lock, [this, &localJobId]() 
        { return !m_isRunning || localJobId < m_currentJobID; });
      }
      if(m_func) {
        m_func(id);
      }
      localJobId += m_threads.size();
      std::lock_guard _(m_finishedMtx);
      if(++m_arrived == m_currentJobID) {
        m_func = nullptr;
        // VLOG("JOB finished: " << m_currentJobID);
        m_finishedCv.notify_one(); // last thread notifies about finished job
      }
    }
  }
  catch(std::exception& ex) {
    VLOG("Thread exception: " << ex.what());
    m_isRunning = false;
    m_jobCv.notify_all();
  }

  uint32_t m_currentJobID = 0, m_arrived = 0;
  bool m_isRunning = true, m_jobFinished = false;
  JobFunc m_func;
  std::mutex m_jobMtx, m_finishedMtx;
  std::condition_variable m_jobCv, m_finishedCv;
  std::vector< std::thread > m_threads;
};

#endif // THREADING_HPP
