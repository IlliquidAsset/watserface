from facefusion.program import create_program
from facefusion.program_helper import validate_args
from argparse import _SubParsersAction
import sys

def check_parser(program, name=""):
    for action in program._actions:
        if action.default and action.choices:
            if isinstance(action.default, list):
                for d in action.default:
                    if d not in action.choices:
                        print(f"[{name}] Action {action.dest} failed: {d} not in {action.choices}")
            elif action.default not in action.choices:
                print(f"[{name}] Action {action.dest} failed: {action.default} not in {action.choices}")
        
        if isinstance(action, _SubParsersAction):
            for sub_name, sub_program in action._name_parser_map.items():
                check_parser(sub_program, name + " " + sub_name)

program = create_program()
print(f"Validate args: {validate_args(program)}")
check_parser(program)
