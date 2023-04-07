import sys
import tempfile

import mlflow


def prevent_logging(force=False, debug=True):
    """ Prevents MLFlow from logging if debugging or force flag is set

    :param force: Force disable logging
    :param debug: If True, logging will be disabled if debugging is detected
    (default: True)
    """
    get_trace = getattr(sys, 'gettrace', None)
    if (get_trace is not None and get_trace() and not debug) or force:
        temp_dir = tempfile.TemporaryDirectory()
        print("Debugging detected or force flag set. MLFlow won't log.")
        mlflow.set_tracking_uri(temp_dir.name)
    else:
        print("MLFlow will log to the default location")
