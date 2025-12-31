#!/usr/bin/env python3
import os
import sys
import subprocess

def main():
    # Set the environment variable required for Pydantic/PyO3 compatibility on some systems
    os.environ['PYO3_USE_ABI3_FORWARD_COMPATIBILITY'] = '1'
    
    # Construct the command to run watserface.py using the current Python interpreter
    cmd = [sys.executable, '-u', 'watserface.py', 'run'] + sys.argv[1:]
    
    try:
        # Run watserface.py and wait for it to complete
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == "__main__":
    main()