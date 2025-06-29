import time
import os
import sys

_print_once_seen = set()

def print_once(*args, **kwargs):
    msg = " ".join(map(str, args))
    if msg not in _print_once_seen:
        _print_once_seen.add(msg)
        print(*args, **kwargs)
        sys.stdout.flush()  # 강제 flush
