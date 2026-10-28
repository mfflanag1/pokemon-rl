"""Hash-bound structured model checkpoints; F32 sources are loaded read-only."""
import json
from pathlib import Path
import re

import jax.numpy as jnp
import numpy as np

from pokemon_full_train.storage import digest, save_checkpoint
from pokemon_opening_search.core import load_frozen as load_f32
from pokemon_reference.feature_spec import CATALOG
from .features import FEATURES
from .network import SHAPES, PARAMETER_COUNT, check_state

MODEL_METADATA = {'model_version': 'f34-structured-policy-v1', 'features': FEATURES,
                  'parameter_count': PARAMETER_COUNT, 'categorical_catalog_hash': CATALOG['content_hash']}


def save(folder, parameters, optimizer, metadata):
    check_state(parameters, optimizer)
    if not isinstance(metadata, dict) or any(k in metadata and metadata[k] != v for k, v in MODEL_METADATA.items()):
        raise ValueError('Structured checkpoint metadata')
    for key in ['cohorts', 'seed', 'next_episode']:
        if type(metadata.get(key)) is not int or metadata[key] < 0:
            raise ValueError('Structured checkpoint coordinate: ' + key)
    if metadata['seed'] >= 2**32:
        raise ValueError('Structured checkpoint seed range')
    if 'optimizer_step' in metadata and metadata['optimizer_step'] != int(optimizer[2]):
        raise ValueError('Structured checkpoint optimizer identity')
    if not isinstance(metadata.get('input_hash'), str) or not re.fullmatch('[0-9a-f]{64}', metadata['input_hash']):
        raise ValueError('Structured checkpoint input identity')
    return save_checkpoint(folder, parameters, optimizer, {**metadata, **MODEL_METADATA})


def load(folder, expected_metadata=None):
    folder = Path(folder)
    pointer = json.loads((folder / 'latest.json').read_text())
    if (not isinstance(pointer, dict) or set(pointer) != {'file', 'sha256'}
            or not isinstance(pointer['file'], str) or Path(pointer['file']).name != pointer['file']):
        raise ValueError('Structured checkpoint pointer')
    path = folder / pointer['file']
    if digest(path) != pointer['sha256']:
        raise ValueError('Structured checkpoint digest')
    with np.load(path, allow_pickle=False) as z:
        expected = {'metadata'} | {prefix + '__' + key for prefix in ['p', 'm', 'v'] for key in SHAPES}
        if set(z.files) != expected or z['metadata'].dtype != np.uint8 or z['metadata'].ndim != 1:
            raise ValueError('Structured checkpoint inventory')
        metadata = json.loads(z['metadata'].tobytes())
        if any(metadata.get(k) != v for k, v in MODEL_METADATA.items()):
            raise ValueError('Structured checkpoint model identity')
        for key in ['cohorts', 'seed', 'next_episode', 'optimizer_step']:
            if type(metadata.get(key)) is not int or metadata[key] < 0:
                raise ValueError('Structured checkpoint coordinate: ' + key)
        if metadata['optimizer_step'] >= 2**31 or metadata['seed'] >= 2**32:
            raise ValueError('Structured checkpoint counter range')
        if not isinstance(metadata.get('input_hash'), str) or not re.fullmatch('[0-9a-f]{64}', metadata['input_hash']):
            raise ValueError('Structured checkpoint input identity')
        if expected_metadata is not None and any(metadata.get(k) != v for k, v in expected_metadata.items()):
            raise ValueError('Structured checkpoint run identity')
        groups = [{key: z[prefix + '__' + key] for key in SHAPES} for prefix in ['p', 'm', 'v']]
        step = np.int32(metadata['optimizer_step'])
        check_state(groups[0], (groups[1], groups[2], step))
        groups = [{k: jnp.asarray(v) for k, v in group.items()} for group in groups]
    return groups[0], (groups[1], groups[2], jnp.int32(step)), metadata
