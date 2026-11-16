#!/usr/bin/env python3
from pathlib import Path
import os
import sys
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'keyframe-mpl'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from keyframe.web_server import main
if __name__ == '__main__':
    main()
