#!/usr/bin/env bash
REMOTE="$1"
REMOTE_DIR=dl1617-lstm
VENV=dl1617-mnist-venv
EXECUTABLE="$2"
shift
shift

rsync -axv *.py requirements.* "$REMOTE:$REMOTE_DIR/"

RUN2="source $VENV/bin/activate && cd $REMOTE_DIR && python $EXECUTABLE $*; bash"
RUN1="tmux new \"$RUN2\""
echo "$RUN1"
ssh -t "$REMOTE" "$RUN1"
