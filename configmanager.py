"""
a:zak-45
d:20/12/2025
v:1.0.0

Manages configuration settings for the application across different environments.

"""

import os

import sys

PLATFORM = sys.platform.lower()

def root_path(filename):
    """
    Determines the root path of the application based on whether it's running from a compiled binary or in development mode.
    Returns the appropriate root path for accessing application resources, handling different OS structures.

    Args:
        filename (str): The name of the file or directory relative to the application's root.

    Returns:
        str: The absolute path to the specified file or directory.

    Examples:
        >> root_path('data/config.ini')
        '/path/to/app/data/config.ini'

    Handles different execution environments (compiled vs. development) to ensure consistent resource access.
    """

    if compiled():  # Running from a compiled binary (Nuitka, PyInstaller)
        if PLATFORM == "darwin":  # macOS APP structure
            base_path = os.path.dirname(os.path.dirname(sys.argv[0]))  # Contents/
            # Nuitka puts files in the same dir as the binary
            return os.path.join(base_path, "MacOS", filename)
        else:  # Windows/Linux
            if "NUITKA_ONEFILE_PARENT" in os.environ:
                """
                When this env var exist, this mean run from the one-file compressed executable.
                This env not exist when run from the extracted program.
                Expected way to work.
                """
                # Nuitka compressed version extract binaries to "WLEDVideoSync" folder (as set in the GitHub action)
                base_path = os.path.join(os.path.dirname(sys.argv[0]), 'WLEDVideoSync')
            else:
                base_path = os.path.dirname(sys.argv[0])
            return os.path.join(base_path, filename)

    # Running in development mode (not compiled)
    return os.path.join(os.path.dirname(__file__),filename)


def compiled():
    return bool(getattr(sys, 'frozen',False) or '__compiled__' in globals())
