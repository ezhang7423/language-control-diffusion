# type: ignore[attr-defined]
"""Code for efficiently scaling through space, time, and tasks"""

"""
Monkey Patching
"""
# monkey patch loguru to default with colors
import loguru

from lcd.utils.setup import abspath

loguru.logger = loguru.logger.opt(colors=True)


def remove_shm_from_resource_tracker():
    """
    Monkey patch multiprocessing.resource_tracker so SharedMemory won't be tracked
    More details at: https://bugs.python.org/issue38119
    """
    # pylint: disable=protected-access, import-outside-toplevel
    # Ignore linting errors in this bug workaround hack
    from multiprocessing import resource_tracker

    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return None
        return resource_tracker._resource_tracker.register(name, rtype)

    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return None
        return resource_tracker._resource_tracker.unregister(name, rtype)

    resource_tracker.unregister = fix_unregister
    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]


# More details at: https://bugs.python.org/issue38119
remove_shm_from_resource_tracker()

import sys
from importlib import metadata as importlib_metadata

"""
Global variables
"""
from pathlib import Path

# * Feel free to change this path if your data is stored somewhere else
# DATA_PATH = (abspath() / "../../submodules/hulc-data").resolve()
DATA_PATH = (abspath() / "../../submodules/hulc-data").resolve()
HULC_PATH = (abspath() / "../../submodules/hulc-baseline").resolve()
REPO_PATH = (abspath() / "../../").resolve()

# Check these paths exist
assert DATA_PATH.exists(), f"{DATA_PATH=} does not exist"
assert HULC_PATH.exists(), f"{HULC_PATH=} does not exist"
assert REPO_PATH.exists(), f"{REPO_PATH=} does not exist"

"""
Versioning
"""


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
