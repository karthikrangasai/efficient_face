import time
from datetime import datetime
from inspect import isfunction, ismethod
from typing import Callable, Dict

from dateutil import relativedelta


def get_name(function: Callable) -> str:
    fn_name = ""
    if ismethod(function):
        fn_name = f"{function.__self__.__class__.__name__}.{function.__name__}"
    elif isfunction(function):
        fn_name = function.__name__
    return fn_name


def profile(function: Callable, fn_kwargs: Dict = None):
    if fn_kwargs is None:
        fn_kwargs = {}

    fn_name = get_name(function=function)

    print(f"[INFO]: Starting `{fn_name}`.", flush=True)

    dt0 = datetime.fromtimestamp(time.time())
    ret_val = function(**fn_kwargs)
    dt1 = datetime.fromtimestamp(time.time())

    print(f"[INFO]: Finished `{fn_name}`.", flush=True)

    rd = relativedelta.relativedelta(dt1, dt0)
    print(
        "\n"
        f"Total time for `{fn_name}`: "
        f"{rd.years}Y {rd.months}M {rd.days}D {rd.hours}h {rd.minutes}m {rd.seconds}s"
        "\n",
        flush=True,
    )
    return ret_val
