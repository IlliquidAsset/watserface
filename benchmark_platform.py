
import platform
import timeit

def benchmark():
    setup = "import platform"
    stmt = "platform.system()"
    times = timeit.repeat(stmt, setup, repeat=5, number=100000)
    print(f"Uncached platform.system() (100k calls): {min(times):.4f}s")

    setup_cached = """
import platform
_SYSTEM = platform.system()
def is_system():
    return _SYSTEM
"""
    stmt_cached = "is_system()"
    times_cached = timeit.repeat(stmt_cached, setup_cached, repeat=5, number=100000)
    print(f"Cached variable (100k calls): {min(times_cached):.4f}s")

if __name__ == "__main__":
    benchmark()
