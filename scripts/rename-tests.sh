#!/bin/bash

#set -x

REPO_NAME=$(basename "$(git rev-parse --show-toplevel)")
echo "Repository name: $REPO_NAME"

if [ $# -ne 1 ]; then
    echo "Usage:	$0 [test folder]"; exit 1
elif [ ! -d "$1" ]; then
    echo "Error: specified target was not a folder"; exit 1
else
    # Target specified was a folder
    TARGET="$1"
fi

for TEST in $(find "$TARGET" -type f -name '*_test.py' | xargs -0); do
    echo "Processing: $TEST"
    FILE_PATH=$(dirname "$TEST")
    FILE_NAME=$(basename "$TEST")
    STRIPPED="${FILE_NAME//_test.py/.py}"
    echo "  git mv \"${TEST}\" $FILE_PATH/test_\"${STRIPPED%%}\""
    git mv "${TEST}" "$FILE_PATH"/test_"${STRIPPED%%}"
done
