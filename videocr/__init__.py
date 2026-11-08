import os

# Suppress the OCR engine's initialization output before any imports
os.environ.setdefault("GLOG_minloglevel", "3")  # Suppress glog (C++ logging)
os.environ.setdefault("FLAGS_minloglevel", "3")  # glog, which onnxruntime may load
os.environ.setdefault("PADDLEOCR_QUIET", "1")  # kept for the engine adapter

from .api import get_subtitles, save_subtitles_to_file
