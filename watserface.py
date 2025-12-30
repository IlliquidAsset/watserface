#!/usr/bin/env python3

import os
import sys
import traceback

os.environ['OMP_NUM_THREADS'] = '1'

from watserface import core

if __name__ == '__main__':
	try:
		core.cli()
	except SystemExit as e:
		print(f"SystemExit: {e.code}")
		sys.exit(e.code)
	except Exception:
		traceback.print_exc()
		sys.exit(1)
