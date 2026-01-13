import logging
import os
from logging import Logger, basicConfig, getLogger, FileHandler

import watserface.choices
from watserface.common_helper import get_first, get_last
from watserface.types import LogLevel

# Global file handler for persistent logging
LOG_FILE = 'watserface.log'
_file_handler = None

def init(log_level : LogLevel) -> None:
	global _file_handler
	
	# Basic config for console
	basicConfig(format = '[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
	
	package_logger = get_package_logger()
	package_logger.setLevel(watserface.choices.log_level_set.get(log_level))
	
	# Add file handler if not already added
	if not _file_handler:
		try:
			_file_handler = FileHandler(LOG_FILE, mode='a', encoding='utf-8')
			_file_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
			package_logger.addHandler(_file_handler)
		except Exception as e:
			print(f"Failed to initialize file logging: {e}")


def get_package_logger() -> Logger:
	return getLogger('watserface')


def debug(message : str, module_name : str) -> None:
	get_package_logger().debug(create_message(message, module_name))


def info(message : str, module_name : str) -> None:
	get_package_logger().info(create_message(message, module_name))


def warn(message : str, module_name : str) -> None:
	get_package_logger().warning(create_message(message, module_name))


def error(message : str, module_name : str) -> None:
	get_package_logger().error(create_message(message, module_name))


def create_message(message : str, module_name : str) -> str:
	# Handle None message
	if message is None:
		message = "Unknown error"

	module_names = module_name.split('.')
	last_module_name = get_last(module_names)

	# Remove 'watserface.' prefix and just use the module name
	if last_module_name:
		return '[' + last_module_name.upper() + '] ' + message
	return message


def enable() -> None:
	get_package_logger().disabled = False


def disable() -> None:
	get_package_logger().disabled = True
