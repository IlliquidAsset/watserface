from typing import Optional

METADATA =\
{
	'name': 'WatserFace',
	'description': 'Advanced face manipulation and training platform',
	'version': '0.10.0',
	'license': 'OpenRAIL-AS',
	'author': 'IlliquidAsset (based on FaceFusion by Henry Ruhs)',
	'url': 'https://github.com/IlliquidAsset/facefusion'
}


def get(key : str) -> Optional[str]:
	if key in METADATA:
		return METADATA.get(key)
	return None
