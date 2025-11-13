from contextlib import contextmanager
import datetime

from loguru import logger


def log_afunc(log_result=False, message=None):
    """Decorator to log execution time of async functions

    Args:
        log_result: If True, also log the function result
        message: Optional message to include in log statements for additional context
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Use custom message if provided, otherwise use function name
            log_message = message or func.__name__

            logger.info(f"[START FUNC] {log_message}")
            start_time = datetime.datetime.now()
            try:
                result = await func(*args, **kwargs)
                end_time = datetime.datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.info(f"[END FUNC] {log_message}. elapsed_time={duration:.2f}s")
                if log_result:
                    logger.info(f"[RESULT FUNC] {log_message}. {result=}")
                return result
            except Exception as e:
                raise e

        return wrapper

    return decorator


def log_aiter_func(message=None, show_each=False):
    """Decorator to log timing information for async generator functions

    Args:
        message: Optional message to include in log statements for additional context
        measure_each: If True, log time between each iteration

    Usage:
        @time_aiter_func(message="Processing data", measure_each=True)
        async def my_generator():
            yield item
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Use custom message if provided, otherwise use function name
            log_message = message or func.__name__

            logger.info(f"[START ITER] {log_message}")
            start_time = datetime.datetime.now()
            item_count = 0
            last_time = start_time
            iter_timings = []

            try:
                async for item in func(*args, **kwargs):
                    current_time = datetime.datetime.now()
                    time_since_last = (current_time - last_time).total_seconds()
                    iter_timings.append(round(time_since_last, 2))

                    if item_count == 0:
                        logger.info(
                            f"[FIRST_ITEM ITER] {log_message}. time_to_first_item={time_since_last:.2f}s"
                        )

                    item_count += 1
                    last_time = current_time
                    yield item
            except Exception as e:
                logger.exception(f"[ERROR] {log_message}. {e}")
                raise e

            finally:
                end_time = datetime.datetime.now()
                total_duration = (end_time - start_time).total_seconds()

                # Use iter_timings for all timing information
                time_to_first_item = f"{iter_timings[0]:.2f}s" if iter_timings else "0s"

                displayed_timings = iter_timings if show_each else []

                logger.info(
                    f"[END ITER] {log_message}. items_yielded_count={item_count},"
                    + f" total_time={total_duration:.2f}s, time_to_first_item={time_to_first_item},"
                    + f" iter_timings={displayed_timings}"
                )

        return wrapper

    return decorator


@contextmanager
def context_log(message: str):
    """
    Context manager that logs entry and exit messages.

    Args:
        message: The message to log (will be prefixed with [ENTER] and [EXIT])

    Example:
        with context_logger("Starting service"):
            # do something
    """
    logger.info(f"[START] {message}")
    start_time = datetime.datetime.now()

    try:
        yield
    except Exception as e:
        duration = (datetime.datetime.now() - start_time).total_seconds()
        logger.exception(f"[ERROR] {message}. duration={duration:.2f}s. {e}")
        raise
    finally:
        duration = (datetime.datetime.now() - start_time).total_seconds()
        logger.info(f"[EXIT] {message}. duration={duration:.2f}s")
