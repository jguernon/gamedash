#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$DIR/gamedash_venv/bin/activate"
python "$DIR/gamedash.py" "$@"
