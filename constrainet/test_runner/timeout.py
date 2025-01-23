"""
Source: https://stackoverflow.com/a/31667005

"""
import threading
import _thread as thread
import sys
import logging
import multiprocessing


def quit_function(fn_name, name):
    message = f'Method {name!r} took too long (function: {fn_name!r})'
    print(message, file=sys.stderr)
    logging.error(message)
    sys.stderr.flush()  # Python 3 stderr is likely buffered.
    thread.interrupt_main()  # raises KeyboardInterrupt


def exit_after(timeout, name: str = ''):
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(timeout, quit_function, args=[fn.__name__, name])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer


def run_function_with_timeout(fn, args, kwargs, name: str, timeout: float, cleanup_time: float = 1.0):
    """Runs the function in a separate process to be able to forcely kill it
    after violating hard time constraints.

    Returns:
        Tuple (status, result).

    """
    timer_wrapped = exit_after(timeout, name=name)(fn)
    def _with_status_wrapper(*a, __result_list, **kwa):
        try:
            result = timer_wrapped(*a, **kwa)
        except KeyboardInterrupt:
            __result_list[0] = False, None
        __result_list[0] =  True, result

    res = [(False, None)]
    p = multiprocessing.Process(
        target=_with_status_wrapper,
        args=args,
        kwargs={'__result_list': res, **kwargs}
    )
    p.start()
    p.join(timeout + cleanup_time)  # wait
    if p.is_alive():
        message = f'Cannot easily terminate function that took too long: {name!r}'
        print(message, file=sys.stderr)
        sys.stderr.flush()
        logging.error(message)
        p.terminate()
        p.join()
    return res[0]


# if __name__ == '__main__':
#     def fn(n_iter: int):
#         s = 0.0
#         for i in range(n_iter):
#             for j in range(n_iter):
#                 s += i * j
#         return s

#     from time import time

#     before = time()
#     try:
#         result = exit_after(0.95, name='test function')(fn)(n_iter=5000)
#     except KeyboardInterrupt:
#         print('No result')
#     after = time()
#     print(f'Elapsed: {(after - before):.03f}')
