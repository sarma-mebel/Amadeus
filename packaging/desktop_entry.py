"""PyInstaller entry point for the Amadeus desktop app.

Kept separate from the package so the spec has a plain script to analyze;
all real logic lives in amadeus.desktop_webview.
"""

import multiprocessing

from amadeus.desktop_webview import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
