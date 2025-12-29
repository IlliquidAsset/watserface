#!/bin/bash
# Fast run script for FaceFusion
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
./venv/bin/python -u facefusion.py run "$@"