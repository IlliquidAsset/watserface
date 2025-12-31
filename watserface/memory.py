from watserface.common_helper import is_macos, is_windows

if is_windows():
	import ctypes
else:
	import resource


def limit_system_memory(system_memory_limit : int = 1) -> bool:
	if is_macos():
		import platform
		# Auto-cap at 10GB for Apple Silicon if not specified or if potentially too high?
		# PRD says: "Memory limit capped at 10GB for Apple Silicon"
		# The input `system_memory_limit` is likely in GB from settings.
		if platform.machine() == 'arm64':
			# Cap at 10GB for Apple Silicon
			if system_memory_limit > 10:
				system_memory_limit = 10

		# Fix: 1024 ** 6 was likely incorrect (Exabytes).
		# If input is in GB, converting to bytes is * 1024 ** 3.
		# If the original code intended something else, it is likely a bug.
		# Correcting to GB -> Bytes conversion.
		system_memory_limit = system_memory_limit * (1024 ** 3)
	else:
		system_memory_limit = system_memory_limit * (1024 ** 3)
	try:
		if is_windows():
			ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(system_memory_limit), ctypes.c_size_t(system_memory_limit)) #type:ignore[attr-defined]
		else:
			resource.setrlimit(resource.RLIMIT_DATA, (system_memory_limit, system_memory_limit))
		return True
	except Exception:
		return False
