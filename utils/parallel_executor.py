import os
import time
import threading
import concurrent.futures
from typing import Callable, List, Any, Optional, Union
from utils.logger import get_logger

logger = get_logger("ParallelExecutor")


class ThreadSafeRateLimiter:
    """
    Helper class to manage request rates across multiple threads.
    """

    def __init__(self, max_per_minute: int):
        self.interval = 60.0 / max_per_minute
        self.lock = threading.Lock()
        self.last_check = 0.0

    def wait_for_slot(self):
        with self.lock:
            current_time = time.time()
            # Calculate when the next request is allowed
            next_allowed = self.last_check + self.interval
            wait_time = next_allowed - current_time

            if wait_time > 0:
                logger.debug(f"Rate limiter: Waiting {wait_time:.3f}s for slot.")
                time.sleep(wait_time)
                # Update last_check to the time we actually execute (now + wait)
                self.last_check = current_time + wait_time
            else:
                # We are good to go immediately
                self.last_check = current_time


def _worker_wrapper(
        func: Callable,
        args: Union[list, tuple],
        max_retries: int = 0,
        retry_timer: float = 0,
        rate_limiter: Optional[ThreadSafeRateLimiter] = None
) -> Any:
    """
    Internal wrapper to handle unpacking, rate limiting, and retries.
    """
    # 1. Apply Rate Limiting
    if rate_limiter:
        rate_limiter.wait_for_slot()

    attempts = 0
    last_exception = None

    # 2. Retry Logic
    while attempts <= max_retries:
        try:
            # 3. Argument Unpacking (*args)
            return func(*args)
        except Exception as e:
            last_exception = e
            attempts += 1
            if attempts <= max_retries:
                logger.warning(f"Task failed (attempt {attempts}). Retrying in {retry_timer}s... Error: {e}")
                time.sleep(retry_timer)

    if last_exception:
        logger.error(f"Task failed after {attempts} attempts. Final Error: {last_exception}")
    # Return the exception if all retries fail so the main thread knows what happened
    return last_exception


def calculate_worker_count(max_threads: int = 0, data_size: int = 0) -> int:
    """
    Determines the number of threads based on user rules.
    """
    cpu_count = os.cpu_count() or 1
    
    if max_threads == 0:
        count = data_size if data_size > 0 else 1
    elif max_threads == -1:
        count = cpu_count
    elif max_threads < -1:
        count = cpu_count * abs(max_threads)
    else:
        count = max_threads

    logger.info(f"Calculated worker count: {count} (Input max_threads: {max_threads}, Data size: {data_size}, CPU cores: {cpu_count})")
    return count


def parallel_execute(
        target_function: Callable,
        data: List[Any],
        max_threads: int = 0,
        max_req_per_min: Optional[int] = None,
        max_retries: int = 0,
        retry_timer: float = 0
) -> List[Any]:
    """
    Parallelizes the execution of a function over a list of data arguments.

    Args:
        target_function: The function to execute. MUST be thread-safe.
        data: A list of items to pass to the function.
              - If an item is a list/tuple, it is unpacked (*item).
              - If an item is any other type, it is passed as a single argument (wrapped in tuple).
        max_threads:
            0 = One thread per data item (unbounded).
            -1 = Number of CPU cores.
            -x = x * Number of CPU cores.
            >0 = Exact number of threads.
        max_req_per_min: Optional limit on executions per minute.
        max_retries: Number of times to retry a failed task.
        retry_timer: Seconds to wait between retries. Defaults to 0.

    Returns:
        List of results in the same order as the input data.
        If a task failed completely, the Exception object is at that index.
    """
    logger.info(f"Starting parallel execution for {len(data)} items.")
    normalized_data = []
    for item in data:
        if isinstance(item, (list, tuple)):
            normalized_data.append(item)
        else:
            normalized_data.append((item,))

    num_workers = calculate_worker_count(max_threads, len(normalized_data))
    rate_limiter = ThreadSafeRateLimiter(max_req_per_min) if max_req_per_min else None

    results: List[Any] = [None] * len(normalized_data)
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Map futures to their original index to preserve order
        future_to_index = {
            executor.submit(
                _worker_wrapper,
                target_function,
                args,
                max_retries,
                retry_timer,
                rate_limiter
            ): idx
            for idx, args in enumerate(normalized_data)
        }

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
            except Exception as exc:
                logger.error(f"Critical error in ThreadPoolExecutor worker: {exc}")
                results[index] = exc

    duration = time.time() - start_time
    logger.info(f"Parallel execution finished in {duration:.2f}s.")
    return results

# ==========================================
# SAMPLE USAGE (Copy and uncomment to run)
# ==========================================
if __name__ == "__main__":
    def sample_task(x, y):
        # Simulate work
        time.sleep(0.5)
        if x == 2: # Simulate a failure to test retries
            raise ValueError("Simulated error")
        return x + y

    # Data: List of argument tuples
    input_data = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]

    print("Running parallel execution...")
    output = parallel_execute(
        target_function=sample_task,
        data=input_data,
        max_threads=-1,       # Use all CPU cores
        max_req_per_min=600,  # Limit to ~10 req/sec
        max_retries=2,        # Retry failed tasks twice
        retry_timer=0.1
    )

    print("Results:")
    for i, res in enumerate(output):
        # Check for Exceptions to handle failures gracefully
        if isinstance(res, Exception):
            print(f"Index {i}: Task failed with error -> {res}")
        else:
            print(f"Index {i}: Success -> {res}")
