from typing import Optional

METADATA =\
{
	'name': 'WatserFace',
	'description': 'Unified 2.5D Face Synthesis Pipeline',
	'version': '1.0.0',
	'license': 'OpenRAIL-AS',
	'author': 'IlliquidAsset (based on FaceFusion by Henry Ruhs)',
	'url': 'https://github.com/IlliquidAsset/facefusion'
}


def get(key : str) -> Optional[str]:
	if key in METADATA:
		return METADATA.get(key)
	return None
