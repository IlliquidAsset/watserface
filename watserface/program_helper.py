from argparse import ArgumentParser, _ArgumentGroup, _SubParsersAction
from typing import Optional


def find_argument_group(program : ArgumentParser, group_name : str) -> Optional[_ArgumentGroup]:
	for group in program._action_groups:
		if group.title == group_name:
			return group
	return None


def validate_args(program : ArgumentParser) -> bool:
	if validate_actions(program):
		for action in program._actions:
			if isinstance(action, _SubParsersAction):
				for name, sub_program in action._name_parser_map.items():
					if not validate_args(sub_program):
						print(f"Validation Failed in sub-parser: {name}")
						return False
		return True
	print("Validation Failed in main parser")
	return False


def validate_actions(program : ArgumentParser) -> bool:
	for action in program._actions:
		if action.default and action.choices:
			if isinstance(action.default, list):
				if any(default not in action.choices for default in action.default):
					print(f"Validation Failed: List default {action.default} not in choices {action.choices} for {action.dest}")
					return False
			elif action.default not in action.choices:
				print(f"Validation Failed: Default {action.default} not in choices {action.choices} for {action.dest}")
				return False
	return True
