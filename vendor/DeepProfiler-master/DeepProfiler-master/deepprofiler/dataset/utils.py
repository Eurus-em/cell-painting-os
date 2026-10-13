"""Shared utilities: timing, progress reporting, parallel processing, logging."""

import logging
import multiprocessing
import os
import sys
import time

PI = 3.1415926539

# Standard library logging idiom — each module gets its own named logger
# inheriting configuration from the root.  This avoids the duplicate-handler
# problem that the old module-level Logger() instantiation caused on re-import.
logger = logging.getLogger(__name__)


def print_progress(iteration, total, prefix="Progress", suffix="Complete", decimals=1, barLength=50):
    """Print an in-place ASCII progress bar to stdout.

    Args:
        iteration: Current iteration (int, 0-indexed).
        total: Total number of iterations (int).
        prefix: Label printed before the bar.
        suffix: Label printed after the percentage.
        decimals: Decimal places in the percentage display.
        barLength: Character width of the bar.
    """
    if all(t >= 0 for t in [iteration, total, barLength]) and iteration <= total:
        formatStr = "{0:." + str(decimals) + "f}"
        percents = formatStr.format(100 * (iteration / float(total)))
        filledLength = int(round(barLength * iteration / float(total)))
        bar = "#" * filledLength + "-" * (barLength - filledLength)
        sys.stdout.write("\r%s |%s| %s%s %s" % (prefix, bar, percents, "%", suffix)),
        sys.stdout.flush()
        if iteration == total:
            sys.stdout.write("\n")
            sys.stdout.flush()
    elif sum([iteration < 0, total < 0, barLength < 0]) > 1:
        sys.stdout.write("\rError: print_progress() function received multiple negative values.")
        sys.stdout.flush()
    elif iteration < 0:
        sys.stdout.write("\rError: print_progress() function received a negative 'iteration' value.")
        sys.stdout.flush()
    elif total < 0:
        sys.stdout.write("\rError: print_progress() function received a negative 'total' value.")
        sys.stdout.flush()
    elif barLength < 0:
        sys.stdout.write("\rError: print_progress() function received a negative 'barLength' value.")
        sys.stdout.flush()
    elif iteration > total:
        sys.stdout.write("\rError: print_progress() function received an 'iteration' value greater than the 'total' value.")
        sys.stdout.flush()


def check_path(filename):
    """Ensure the parent directory of ``filename`` exists, creating it if needed."""
    path = "/".join(filename.split("/")[0:-1])
    os.makedirs(path, exist_ok=True)


def tic():
    """Return the current wall-clock time in seconds (for use with :func:`toc`)."""
    return time.time()


def toc(msg, beginning):
    """Print elapsed time since ``beginning`` and return the current time.

    Args:
        msg: Label printed before the elapsed time.
        beginning: Start time returned by :func:`tic`.

    Returns:
        Current wall-clock time in seconds.
    """
    end = time.time()
    elapsed = end - beginning
    print(msg, ": {:.2f} secs".format(elapsed))
    return end


class Parallel():
    """Thin wrapper around :class:`multiprocessing.Pool` for map-style parallelism.

    Each call to :meth:`compute` invokes ``operation([item, fixed_args])`` for
    every item in ``data`` across the worker pool.  The fixed args pattern
    lets callers pass configuration or other shared state without using global
    variables.

    Args:
        fixed_args: A single object passed as the second element of every
            worker invocation — typically the experiment config dict.
        numProcs: Number of worker processes.  Defaults to CPU count; clamped
            to ``[1, cpu_count]``.
    """

    def __init__(self, fixed_args, numProcs=None):
        """Create the worker pool with up to ``numProcs`` processes."""
        self.fixed_args = fixed_args
        cpus = multiprocessing.cpu_count()
        if numProcs is None or numProcs > cpus or numProcs < 1:
            numProcs = cpus
        self.pool = multiprocessing.Pool(numProcs)

    def compute(self, operation, data):
        """Run ``operation`` on every element of ``data`` in parallel.

        Args:
            operation: Callable that accepts ``[item, fixed_args]``.
            data: Iterable of items to process.

        Returns:
            List of results in the same order as ``data``.
        """
        iterable = [[d, self.fixed_args] for d in data]
        return self.pool.map(operation, iterable)

    def close(self):
        """Shut down the worker pool, waiting for all jobs to finish."""
        self.pool.close()
        self.pool.join()


# Configure a single stdout handler on the root logger (idempotent — repeated
# calls by subprocess workers are harmless because we guard with hasHandlers).
_root = logging.getLogger()
if not _root.hasHandlers():
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setLevel(logging.DEBUG)
    _handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    _root.addHandler(_handler)
    _root.setLevel(logging.INFO)
