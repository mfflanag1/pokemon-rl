"""Hash-bound accepted teacher actions and deterministic game-uniform sampling."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np

from pokemon_structured_policy.features import encode_packed
from pokemon_structured_policy.learning import validate_features

FIELDS = frozenset({'features', 'mask', 'actions'})
LABEL_FIELDS = frozenset({'row', 'step_index', 'request_id', 'side', 'action', 'submission_index',
    'player_sha256', 'history_sha256', 'features_sha256', 'mask_sha256', 'external_input_sha256'})
SAMPLING_NAMESPACE = 0x46333544


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def validate_demo(batch):
    """Only unpadded supervised action inputs are accepted, never PPO targets."""
    if not isinstance(batch, dict) or set(batch) != FIELDS:
        raise ValueError('Demonstration input inventory: only features, mask, actions')
    x, mask, actions = (np.asarray(batch[k]) for k in ('features', 'mask', 'actions'))
    if actions.ndim != 1 or len(actions) == 0 or actions.dtype != np.int32:
        raise ValueError('Demonstration action shape/dtype')
    n = len(actions)
    validate_features(x, np.ones(n, bool))
    if mask.shape != (n, 14) or mask.dtype != np.bool_ or not mask.any(1).all():
        raise ValueError('Demonstration legal mask')
    if not ((actions >= 0) & (actions < 14)).all() or not mask[np.arange(n), actions].all():
        raise ValueError('Demonstration accepted action must be legal')
    return {'features': x, 'mask': mask, 'actions': actions}


def _uint(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= int(value) < 2**32:
        raise ValueError('Unsigned demonstration ' + name)
    return int(value)


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _read(path):
    return json.loads(path.read_text())


def _safe_file(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('Unsafe demonstration artifact path')
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or path.is_symlink():
        raise ValueError('Missing or redirected demonstration artifact')
    return path


class Dataset:
    """Immutable complete-game corpus. Scientific data enter through ``load``."""
    def __init__(self, games, *, split, manifest_sha256=None, terminal_returns=None, team_keys=()):
        if split not in ('train', 'development') or not isinstance(games, list) or not games:
            raise ValueError('Demonstration role or empty corpus')
        self.split, self.manifest_sha256 = split, manifest_sha256
        if any(not _hash(key) for key in team_keys):
            raise ValueError('Demonstration team identity')
        self.team_keys = frozenset(team_keys)
        ids, batches, offsets = [], [], [0]
        for game_id, batch in games:
            if not isinstance(game_id, str) or not game_id or game_id in ids:
                raise ValueError('Duplicate or invalid demonstration game ID')
            b = validate_demo(batch)
            ids.append(game_id); batches.append(b); offsets.append(offsets[-1] + len(b['actions']))
        self.game_ids = tuple(ids)
        if terminal_returns is None:
            terminal_returns = [None] * len(ids)
        if (not isinstance(terminal_returns, (list, tuple)) or len(terminal_returns) != len(ids)
                or any(v is not None and (isinstance(v, (bool, np.bool_))
                    or not isinstance(v, (int, float, np.integer, np.floating))
                    or not np.isfinite(v) or not 0 <= v <= 1) for v in terminal_returns)):
            raise ValueError('Diagnostic terminal outcome inventory')
        self.terminal_returns = tuple(None if v is None else float(v) for v in terminal_returns)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.arrays = {k: np.concatenate([b[k] for b in batches]) for k in FIELDS}
        self.rows = int(offsets[-1])
        for array in [self.offsets, *self.arrays.values()]:
            array.setflags(write=False)

    def sample_with_indices(self, *, seed, cohort, update, rows=256):
        """Local SeedSequence prevents demonstration sampling from touching PPO RNG."""
        coordinates = [_uint(v, k) for k, v in [('seed', seed), ('cohort', cohort), ('update', update)]]
        if type(rows) is not int or rows <= 0:
            raise ValueError('Positive demonstration sample size required')
        rng = np.random.default_rng(np.random.SeedSequence([SAMPLING_NAMESPACE, *coordinates]))
        games = rng.integers(0, len(self.game_ids), size=rows)
        indices = np.asarray([self.offsets[g] + rng.integers(0, self.offsets[g + 1] - self.offsets[g])
                              for g in games], dtype=np.int64)
        return {k: v[indices] for k, v in self.arrays.items()}, indices

    def sample(self, *, seed, cohort, update, rows=256):
        return self.sample_with_indices(seed=seed, cohort=cohort, update=update, rows=rows)[0]

    @classmethod
    def load(cls, folder, *, expected_input_hash, expected_schedule, catalog,
             split='train', quarantine=(), expected_manifest_sha256=None):
        """Admit every scheduled train/probe game before selecting a declared role.

        File hashes bind the recorded evidence. Every label is independently
        matched to the teacher's accepted submission and re-encoded local view.
        The frozen split-aware Node worker enforces actual whole-team partitions;
        this loader also rejects conflicting role claims and quarantine keys.
        """
        root = Path(folder)
        report_path = _safe_file(root, 'report.json')
        manifest_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
        if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
            raise ValueError('Demonstration manifest digest')
        report = _read(report_path)
        if (report.get('schema_version') != 'f35-teacher-data-v1' or report.get('status') != 'passed'
                or not _hash(expected_input_hash) or report.get('input_hash') != expected_input_hash
                or report.get('schedule') != expected_schedule):
            raise ValueError('Demonstration manifest status/input/schedule identity')
        if not isinstance(expected_schedule, list) or not expected_schedule:
            raise ValueError('Nonempty complete demonstration schedule required')
        expected_ids = [s.get('game_id') for s in expected_schedule]
        games = report.get('games')
        if (not isinstance(games, list) or len(expected_ids) != len(set(expected_ids))
                or [g.get('game_id') for g in games] != expected_ids):
            raise ValueError('Incomplete or duplicate demonstration game schedule')
        selected, terminal_returns, selected_team_keys, seen_directories = [], [], set(), set()
        for expected, game in zip(expected_schedule, games):
            if any(game.get(k) != v for k, v in expected.items()):
                raise ValueError('Demonstration game schedule mismatch')
            role, side = game.get('split'), game.get('external_side')
            if role not in ('train', 'development') or side not in ('p1', 'p2'):
                raise ValueError('Demonstration game role/teacher side')
            directory = game.get('directory')
            if not isinstance(directory, str) or directory in seen_directories:
                raise ValueError('Duplicate demonstration game directory')
            seen_directories.add(directory)
            game_root = root / directory
            if Path(directory).is_absolute() or '..' in Path(directory).parts or not game_root.resolve().is_relative_to(root.resolve()):
                raise ValueError('Unsafe demonstration game directory')
            files = game.get('files')
            required = {'result.json', 'transcript.jsonl', 'labels.json', 'demonstrations.npz'}
            if not isinstance(files, dict) or not required.issubset(files):
                raise ValueError('Demonstration evidence file inventory')
            for relative, expected_hash in files.items():
                path = _safe_file(game_root, relative)
                if not _hash(expected_hash) or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                    raise ValueError('Demonstration evidence digest: ' + relative)
            result = _read(game_root / 'result.json')
            provenance, outcome = result.get('provenance'), result.get('outcome')
            coordinates = ('run_seed', 'episode_index', 'swap', 'fp_seed')
            if any(k in expected and result.get(k) != expected[k] for k in coordinates):
                raise ValueError('Demonstration result schedule coordinates')
            if 'run_seed' in expected:
                expected_id = (f"f2:{role}:{expected['run_seed']}:{expected['episode_index']}:"
                               + ('swap' if expected['swap'] else 'base'))
                scenario = provenance.get('scenario', {}) if isinstance(provenance, dict) else {}
                if (game['game_id'] != expected_id or provenance.get('episode_id') != expected_id
                        or any(scenario.get(k) != expected[k] for k in ('run_seed', 'episode_index', 'swap'))
                        or scenario.get('partition') != role):
                    raise ValueError('Demonstration official schedule provenance')
            if (result.get('status') != 'completed' or result.get('split') != role
                    or result.get('external_side') != side or result.get('game_id') != game['game_id']
                    or provenance != game.get('provenance') or outcome != game.get('outcome')
                    or not isinstance(outcome, dict) or outcome.get('kind') not in ('terminal', 'administrative_cutoff')):
                raise ValueError('Demonstration game completion/provenance')
            if (not isinstance(provenance, dict) or provenance.get('split') != role
                    or not isinstance(provenance.get('teams'), list) or len(provenance['teams']) != 2
                    or any(t.get('partition') != role or not _hash(t.get('team_key')) or t['team_key'] in quarantine
                           for t in provenance['teams'])):
                raise ValueError('Demonstration train/development partition or quarantine violation')
            rewards = outcome.get('rewards')
            if (not isinstance(rewards, dict) or set(rewards) != {'p1', 'p2'}
                    or any(type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 1
                           for v in rewards.values())
                    or (outcome['kind'] == 'terminal' and sum(rewards.values()) != 1)):
                raise ValueError('Demonstration scalar outcome inventory')
            labels_doc = _read(game_root / 'labels.json')
            if (labels_doc.get('schema_version') != 'f35-teacher-labels-v1'
                    or labels_doc.get('game_id') != game['game_id'] or labels_doc.get('split') != role
                    or labels_doc.get('external_side') != side):
                raise ValueError('Demonstration label identity')
            labels = labels_doc.get('labels')
            with np.load(game_root / 'demonstrations.npz', allow_pickle=False) as archive:
                if set(archive.files) != FIELDS | {'step_indices', 'request_ids'}:
                    raise ValueError('Demonstration archive input inventory')
                arrays = {k: archive[k] for k in archive.files}
            b = validate_demo({k: arrays[k] for k in FIELDS})
            n = len(b['actions'])
            if not isinstance(labels, list) or len(labels) != n or game.get('accepted_labels') != n:
                raise ValueError('Demonstration accepted-label count')
            for name in ('step_indices', 'request_ids'):
                if arrays[name].shape != (n,) or arrays[name].dtype != np.int32 or np.any(arrays[name] < 0):
                    raise ValueError('Demonstration label index dtype/range')
            frames = [_read_line(line) for line in (game_root / 'transcript.jsonl').read_text().splitlines()]
            steps = result.get('steps')
            if not isinstance(steps, list) or len(frames) < len(steps):
                raise ValueError('Demonstration transcript/step inventory')
            accepted, teacher_attempts, wait_calls, decision_count = [], 0, 0, 0
            rejected = {'p1': 0, 'p2': 0}
            if (len(frames) not in (len(steps) + 1, len(steps) + 2)
                    or frames[-1].get('frame', {}).get('outcome') != outcome):
                raise ValueError('Demonstration final transcript/outcome linkage')
            for index, step in enumerate(steps):
                if step.get('index') != index:
                    raise ValueError('Demonstration step order')
                frame = frames[index].get('frame', {})
                players = frame.get('players')
                if (not isinstance(players, dict) or set(players) != {'p1', 'p2'}
                        or frame.get('outcome') is not None or frame.get('split') != role
                        or frame.get('provenance') != provenance):
                    raise ValueError('Demonstration pre-action frame inventory')
                for seat, player in players.items():
                    mask = player.get('mask')
                    if (player.get('side') != seat or not isinstance(mask, list) or len(mask) != 14
                            or any(type(v) is not bool for v in mask)
                            or type(player.get('request_id')) is not int or player['request_id'] < 0):
                        raise ValueError('Demonstration active/wait player request')
                active = {seat for seat, player in players.items() if any(player['mask'])}
                actions, request_ids = step.get('actions'), step.get('request_ids')
                submitted = step.get('submitted')
                if (not active or not isinstance(actions, dict) or set(actions) != active
                        or not isinstance(request_ids, dict) or set(request_ids) != active
                        or not isinstance(submitted, list) or len(submitted) != len(active)
                        or {sub.get('side') for sub in submitted} != active
                        or any(request_ids[seat] != players[seat]['request_id'] for seat in active)):
                    raise ValueError('Demonstration active/wait submission inventory')
                teacher = step.get('external')
                if side in active:
                    if (not isinstance(teacher, dict) or type(teacher.get('action')) is not int
                            or teacher['action'] != actions[side] or teacher.get('request_id') != request_ids[side]):
                        raise ValueError('Demonstration teacher active decision linkage')
                    teacher_sub = next(sub for sub in submitted if sub['side'] == side)
                    expected_ack = {'acknowledged': True, 'request_id': request_ids[side],
                                    'accepted': teacher_sub.get('accepted')}
                    ack = step.get('teacher_feedback')
                    if (not isinstance(ack, dict) or set(ack) != set(expected_ack)
                            or type(ack.get('acknowledged')) is not bool or type(ack.get('accepted')) is not bool
                            or type(ack.get('request_id')) is not int or ack != expected_ack):
                        raise ValueError('Demonstration exact teacher feedback acknowledgement')
                    teacher_attempts += 1
                else:
                    if teacher != {'waiting': True} or 'teacher_feedback' in step:
                        raise ValueError('Demonstration waiting teacher must have no action or feedback')
                    wait_calls += 1
                if (step.get('external_phase') != players[side].get('phase')
                        or step.get('external_input_sha256') != json_hash(frames[index]['external'])):
                    raise ValueError('Demonstration teacher phase/input linkage')
                decision_count += len(active)
                for sub_index, sub in enumerate(submitted):
                    submitted_side = sub.get('side')
                    feedback = sub.get('feedback')
                    if (type(sub.get('accepted')) is not bool or submitted_side not in ('p1', 'p2')
                            or type(sub.get('action')) is not int or not 0 <= sub['action'] < 14
                            or type(sub.get('request_id')) is not int or sub['request_id'] < 0
                            or not isinstance(sub.get('command'), str) or not sub['command']
                            or step.get('actions', {}).get(submitted_side) != sub['action']
                            or step.get('request_ids', {}).get(submitted_side) != sub['request_id']
                            or not isinstance(feedback, list)
                            or (sub['accepted'] and feedback)
                            or (not sub['accepted'] and (not feedback or any(not isinstance(f, str)
                                or not f.startswith('[Unavailable choice]') for f in feedback)))):
                        raise ValueError('Demonstration submission acceptance/feedback missing or invalid')
                    if not players[submitted_side]['mask'][sub['action']]:
                        raise ValueError('Demonstration submitted action outside advertised mask')
                    if not sub['accepted']:
                        rejected[submitted_side] += 1
                    if sub.get('side') == side and sub['accepted']:
                        accepted.append((index, sub_index))
            if (len(accepted) != n or result.get('accepted_labels') != n
                    or result.get('foul_play_decisions') != teacher_attempts or result.get('decisions') != decision_count
                    or result.get('rejected_actions') != rejected or teacher_attempts != n + rejected[side]
                    or teacher_attempts + wait_calls != len(steps)):
                raise ValueError('Accepted/rejected/wait teacher decision accounting mismatch')
            for row, (label, (step_index, submission_index)) in enumerate(zip(labels, accepted)):
                if (not isinstance(label, dict) or set(label) != LABEL_FIELDS
                        or any(type(label[k]) is not int or label[k] < 0
                               for k in ('row', 'step_index', 'request_id', 'action', 'submission_index'))):
                    raise ValueError('Demonstration label metadata inventory')
                step = steps[step_index]
                bundle = frames[step_index]
                frame = bundle['frame']
                player, history = frame['players'][side], frame['observable_history'][side]
                action, request_id = int(b['actions'][row]), int(arrays['request_ids'][row])
                sub = step['submitted'][submission_index]
                if (label['row'] != row or label['step_index'] != step_index or label['submission_index'] != submission_index
                        or label['side'] != side or label['action'] != action or label['request_id'] != request_id
                        or int(arrays['step_indices'][row]) != step_index or player.get('side') != side
                        or history.get('side') != side or player.get('request_id') != request_id
                        or frame.get('split') != role or frame.get('provenance') != provenance
                        or step.get('actions', {}).get(side) != action or step.get('request_ids', {}).get(side) != request_id
                        or sub.get('action') != action or sub.get('request_id') != request_id or not sub.get('command')):
                    raise ValueError('Demonstration accepted-action/request alignment')
                encoded, mask = encode_packed(player, catalog, history)
                expected_hashes = {'player_sha256': json_hash(player), 'history_sha256': json_hash(history),
                    'features_sha256': array_hash(encoded), 'mask_sha256': array_hash(mask),
                    'external_input_sha256': json_hash(bundle['external'])}
                if (any(label[k] != v for k, v in expected_hashes.items())
                        or step.get('external_input_sha256') != expected_hashes['external_input_sha256']
                        or not np.array_equal(encoded, b['features'][row]) or not np.array_equal(mask, b['mask'][row])):
                    raise ValueError('Demonstration teacher-local feature/evidence alignment')
            if role == split:
                selected.append((game['game_id'], b))
                terminal_returns.append(rewards[side] if outcome['kind'] == 'terminal' else None)
                selected_team_keys.update(t['team_key'] for t in provenance['teams'])
        return cls(selected, split=split, manifest_sha256=manifest_sha, terminal_returns=terminal_returns,
                   team_keys=selected_team_keys)


def _read_line(line):
    if not line.strip():
        raise ValueError('Empty demonstration transcript record')
    return json.loads(line)
