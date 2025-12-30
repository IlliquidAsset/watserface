from typing import Optional

import gradio

import watserface.choices
from watserface import wording
from watserface.uis.core import register_ui_component

BENCHMARK_RESOLUTIONS_CHECKBOX_GROUP : Optional[gradio.CheckboxGroup] = None
BENCHMARK_CYCLE_COUNT_SLIDER : Optional[gradio.Button] = None


def render() -> None:
	global BENCHMARK_RESOLUTIONS_CHECKBOX_GROUP
	global BENCHMARK_CYCLE_COUNT_SLIDER

	BENCHMARK_RESOLUTIONS_CHECKBOX_GROUP = gradio.CheckboxGroup(
		label = wording.get('uis.benchmark_resolutions_checkbox_group'),
		choices = watserface.choices.benchmark_resolutions,
		value = watserface.choices.benchmark_resolutions
	)
	BENCHMARK_CYCLE_COUNT_SLIDER = gradio.Slider(
		label = wording.get('uis.benchmark_cycle_count_slider'),
		value = 5,
		step = 1,
		minimum = min(watserface.choices.benchmark_cycle_count_range),
		maximum = max(watserface.choices.benchmark_cycle_count_range)
	)
	register_ui_component('benchmark_resolutions_checkbox_group', BENCHMARK_RESOLUTIONS_CHECKBOX_GROUP)
	register_ui_component('benchmark_cycle_count_slider', BENCHMARK_CYCLE_COUNT_SLIDER)
