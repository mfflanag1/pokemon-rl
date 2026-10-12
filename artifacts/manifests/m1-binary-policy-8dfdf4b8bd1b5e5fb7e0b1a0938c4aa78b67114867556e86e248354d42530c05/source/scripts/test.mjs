import {spawnSync} from 'node:child_process';

for (const [command, args, extraEnv] of [
  [process.execPath, ['--test', '--test-concurrency=1', 'tests/reference/*.test.mjs', 'tests/contracts/*.test.mjs', 'tests/corpus/*.test.mjs', 'tests/episodes/*.test.mjs', 'tests/features/*.test.mjs', 'tests/mechanics/*.test.mjs', 'tests/dataset/*.test.mjs', 'tests/slice/*.test.mjs', 'tests/jax-slice/*.test.mjs'], {}],
  [process.env.CONTRACT_PYTHON ?? '.venv/bin/python', ['-m', 'unittest', 'discover', '-s', 'tests/contracts', '-p', 'test_*.py'], {PYTHONPATH: 'python'}],
  [process.env.CONTRACT_PYTHON ?? '.venv/bin/python', ['-m', 'unittest', 'discover', '-s', 'tests/corpus', '-p', 'test_*.py'], {PYTHONPATH: 'python'}],
  [process.env.CONTRACT_PYTHON ?? '.venv/bin/python', ['-m', 'unittest', 'discover', '-s', 'tests/episodes', '-p', 'test_*.py'], {PYTHONPATH: 'python'}],
  [process.env.CONTRACT_PYTHON ?? '.venv/bin/python', ['-m', 'unittest', 'discover', '-s', 'tests/features', '-p', 'test_*.py'], {PYTHONPATH: 'python'}],
  [process.env.CONTRACT_PYTHON ?? '.venv/bin/python', ['-m', 'unittest', 'discover', '-s', 'tests/mechanics', '-p', 'test_*.py'], {PYTHONPATH: 'python'}],
  [process.env.CONTRACT_PYTHON ?? '.venv/bin/python', ['-m', 'unittest', 'discover', '-s', 'tests/dataset', '-p', 'test_*.py'], {PYTHONPATH: 'python'}],
  [process.env.CONTRACT_PYTHON ?? '.venv/bin/python', ['-m', 'unittest', 'discover', '-s', 'tests/learner', '-p', 'test_*.py'], {PYTHONPATH: 'python'}],
  [process.env.CONTRACT_PYTHON ?? '.venv/bin/python', ['-m', 'unittest', 'discover', '-s', 'tests/slice', '-p', 'test_*.py'], {PYTHONPATH: 'python'}],
  [process.env.CONTRACT_PYTHON ?? '.venv/bin/python', ['-m', 'unittest', 'discover', '-s', 'tests/jax-slice', '-p', 'test_*.py'], {PYTHONPATH: 'python'}],
  [process.env.CONTRACT_PYTHON ?? '.venv/bin/python', ['-m', 'unittest', 'discover', '-s', 'tests/jax-learner', '-p', 'test_*.py'], {PYTHONPATH: 'python'}],
  [process.env.CONTRACT_PYTHON ?? '.venv/bin/python', ['-m', 'unittest', 'discover', '-s', 'tests/fixed-loop', '-p', 'test_*.py'], {PYTHONPATH: 'python'}],
  [process.env.CONTRACT_PYTHON ?? '.venv/bin/python', ['-m', 'unittest', 'discover', '-s', 'tests/binary-policy', '-p', 'test_*.py'], {PYTHONPATH: 'python'}],
]) {
  const result = spawnSync(command, args, {stdio: 'inherit', env: {...process.env, ...extraEnv}});
  if (result.error) throw new Error(`Cannot run ${command}; see docs/R2_CONTRACT.md setup instructions`, {cause: result.error});
  if (result.status !== 0) process.exit(result.status ?? 1);
}
