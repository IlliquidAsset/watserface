from facefusion.program import create_program
from facefusion.program_helper import validate_args
import sys

# Mock sys.argv
sys.argv = ['watserface.py', 'run']

program = create_program()
print(f"Validate args with 'run': {validate_args(program)}")