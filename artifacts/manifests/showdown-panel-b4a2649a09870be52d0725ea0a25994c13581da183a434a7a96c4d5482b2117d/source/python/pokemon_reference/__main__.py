"""Validate artifact JSONL from a file or stdin, and emit canonical JSONL."""

import sys
from .contract import canonical, load_artifact


def main():
    sys.stdout.reconfigure(encoding='utf-8', newline='\n')
    stream = open(sys.argv[1], 'rb') if len(sys.argv) == 2 else sys.stdin.buffer
    try:
        for line in stream:
            if line.strip():
                sys.stdout.write(canonical(load_artifact(line)) + '\n')
    finally:
        if stream is not sys.stdin.buffer:
            stream.close()


if __name__ == '__main__':
    main()
