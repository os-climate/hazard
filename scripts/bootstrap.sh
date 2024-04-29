#!/usr/bin/env bash

### Script to bootstrap the OS-Climate DevOps environment ###

set -eu -o pipefail
# set -xv

### Variables ###

SOURCE_FILE="bootstrap.yaml"
WGET_URL="https://raw.githubusercontent.com/os-climate/devops-toolkit/main/.github/workflows/$SOURCE_FILE"
AUTOMATION_BRANCH="update-devops-tooling"
DEVOPS_DIR=".devops"

### Checks ###

GIT_CMD=$(which git)
if [ ! -x "$GIT_CMD" ]; then
    echo "GIT command was NOT found in PATH"; exit 1
fi

WGET_CMD=$(which wget)
if [ ! -x "$WGET_CMD" ]; then
    echo "WGET command was NOT found in PATH"; exit 1
fi

MKTEMP_CMD=$(which mktemp)
if [ ! -x "$MKTEMP_CMD" ]; then
    echo "MKTEMP command was NOT found in PATH"; exit 1
fi

SHELL_SCRIPT=$(mktemp -t script-XXXXXXXX.sh)

### Functions ###

change_dir_error() {
    echo "Could not change directory"; exit 1
}

check_for_local_branch() {
    BRANCH="$1"
    git show-ref --quiet refs/heads/"$BRANCH"
    return $?
}

check_for_remote_branch() {
    BRANCH="$1"
    git ls-remote --exit-code --heads origin "$BRANCH"
    return $?
}

cleanup_on_exit() {
    # Remove PR branch, if it exists
    echo "Cleaning up on exit: bootstrap.sh"
    echo "Swapping from temporary branch to: $HEAD_BRANCH"
    git checkout main > /dev/null 2>&1
    if (check_for_local_branch "$AUTOMATION_BRANCH"); then
        echo "Removing temporary local branch: $AUTOMATION_BRANCH"
        git branch -d "$AUTOMATION_BRANCH" > /dev/null 2>&1
    fi
    if [ -f "$SHELL_SCRIPT" ]; then
        echo "Removing temporary shell code"
        rm "$SHELL_SCRIPT"
    fi
    if [ -d "$DEVOPS_DIR" ]; then
        echo "Removed local copy of devops repository"
        rm -Rf "$DEVOPS_DIR"
    fi
}
trap cleanup_on_exit EXIT

### Main script entry point

# Get organisation and repository name
# git config --get remote.origin.url
# git@github.com:ModeSevenIndustrialSolutions/test-bootstrap.git
URL=$(git config --get remote.origin.url)

# Take the above and store it converted as ORG_AND_REPO
# e.g. ModeSevenIndustrialSolutions/test-bootstrap
ORG_AND_REPO=${URL/%.git}
ORG_AND_REPO=${ORG_AND_REPO//:/ }
ORG_AND_REPO=$(echo "$ORG_AND_REPO" | awk '{ print $2 }')
HEAD_BRANCH=$("$GIT_CMD" rev-parse --abbrev-ref HEAD)
REPO_DIR=$(git rev-parse --show-toplevel)
# Change to top-level of GIT repository
CURRENT_DIR=$(pwd)
if [ "$REPO_DIR" != "$CURRENT_DIR" ]; then
    echo "Changing directory to: $REPO_DIR"
    cd "$REPO_DIR" || change_dir_error
fi

# Get latest copy of bootstrap workflow
if [ -f "$SOURCE_FILE" ]; then
    echo "Removing existing copy of: $SOURCE_FILE"
    rm "$SOURCE_FILE"
fi
echo "Pulling latest DevOps bootstrap workflow from:"
echo "  $WGET_URL"
"$WGET_CMD" -q "$WGET_URL"

# The section below extracts shell code from the YAML file
echo "Extracting shell code from: $SOURCE_FILE"
EXTRACT="false"
while read -r LINE; do
    if [ "$LINE" = "#SHELLCODESTART" ]; then
        EXTRACT="true"
        SHELL_SCRIPT=$(mktemp -t script-XXXXXXXX.sh)
        touch "$SHELL_SCRIPT"
        chmod a+x "$SHELL_SCRIPT"
        echo "Creating shell script: $SHELL_SCRIPT"
        echo "#!/bin/sh" > "$SHELL_SCRIPT"
    fi
    if [ "$EXTRACT" = "true" ]; then
        echo "$LINE" >> "$SHELL_SCRIPT"
        if [ "$LINE" = "#SHELLCODEEND" ]; then
            break
        fi
    fi
done < "$SOURCE_FILE"

echo "Running extracted shell script code"
# https://www.shellcheck.net/wiki/SC1090
# Shell code executed is temporary and cannot be checked by linting
# shellcheck disable=SC1090
. "$SHELL_SCRIPT"
