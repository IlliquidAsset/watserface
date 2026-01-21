import sys
import importlib
import traceback

files_to_check = [
    'watserface.processors.choices',
    'watserface.studio.state',
    'watserface.studio.orchestrator',
    'watserface.training.trainers.xseg',
    'watserface.training.trainers.identity',
    'watserface.uis.layouts.studio'
]

print("--- Checking Syntax and Imports ---")
for module_name in files_to_check:
    try:
        print(f"Checking {module_name}...", end=" ")
        importlib.import_module(module_name)
        print("OK")
    except Exception as e:
        print("FAIL")
        traceback.print_exc()
        sys.exit(1)

print("\n--- Checking UI Construction ---")
try:
    from watserface.uis.layouts import studio
    print("Building Studio UI...", end=" ")
    layout = studio.render()
    print("OK")
except Exception as e:
    print("FAIL")
    traceback.print_exc()
    sys.exit(1)

print("\nAll integrity checks passed.")
