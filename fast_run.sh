#!/bin/bash
# Fast run script for WatserFace
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
./venv/bin/python -u watserface.py run "$@"