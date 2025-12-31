from typing import Optional

METADATA =\
{
	'name': 'WatserFace',
	'description': '',
	'version': '0.16.0',
	'license': 'OpenRAIL-AS',
	'author': 'IlliquidAsset (based on FaceFusion by Henry Ruhs)',
	'url': 'https://github.com/IlliquidAsset/facefusion'
}


def get(key : str) -> Optional[str]:
	if key in METADATA:
		return METADATA.get(key)
	return None
