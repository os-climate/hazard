#!/bin/bash

# set -x

THIS_SCRIPT=$(basename "$0")
echo "This script: $SELF"

TEMPLATE_NAME=osc-python-template
ALT_TEMPLATE_NAME="${TEMPLATE_NAME//-/_}"

### Shared functions

# Renames files/folders containing template name
rename_object() {
    if [ $# -ne 1 ]; then
        echo "Function requires an argumeent: rename_object [filesystem object]"; exit 1
    else
        FS_OBJECT="$1"
    fi
    # Function take a filesystem object as a single argument
    FS_OBJECT="$1"
    OBJECT_PATH=$(dirname "$FS_OBJECT")
    OBJECT_NAME=$(basename "$FS_OBJECT")

    # Check if filesystem object contains template name
    if [[ ! "$OBJECT_NAME" == *"$TEMPLATE_NAME"* ]]; then
        # Nothing to do; abort early
        return
    else
        NEW_NAME="${OBJECT_NAME//$TEMPLATE_NAME/$REPO_NAME}"
    fi
    if [[ ! "$OBJECT_NAME" == *"$ALT_TEMPLATE_NAME"* ]]; then
        # Nothing to do; abort early
        return
    else
        NEW_NAME="${OBJECT_NAME//$ALT_TEMPLATE_NAME/$ALT_REPO_NAME}"
    fi

    # Perform the renaming operation
    if [ -d "$FS_OBJECT" ]; then
        echo "Renaming folder: $FS_OBJECT"
    elif  [ -f "$FS_OBJECT" ]; then
        echo "Renaming file: $FS_OBJECT"
    elif [ -L "$FS_OBJECT" ]; then
        echo "Renaming symlink: $FS_OBJECT"
    fi
    git mv "$OBJECT_PATH/$OBJECT_NAME" "$OBJECT_PATH/$NEW_NAME"
}

# Checks file content for template name and replaces matching strings
file_content_substitution() {
    if [ $# -ne 1 ]; then
        echo "Function requires an argument: file_content_substitution [filename]"; exit 1
    else
        FILENAME="$1"
    fi

    # Do not modify self!
    BASE_FILENAME=$(basename "$FILENAME")
    if [ "$BASE_FILENAME" = "$THIS_SCRIPT" ]; then
        echo "Skipping self: $THIS_SCRIPT"
        return
    fi

    COUNT=0
    if (grep "$TEMPLATE_NAME" "$FILENAME" > /dev/null 2>&1); then
        MATCHES=$(grep -c "$TEMPLATE_NAME" "$FILENAME")
        if [ "$MATCHES" -eq 1 ]; then
            echo "1 content substitution required: $FILENAME (dashes)"
            COUNT=$((COUNT++))
        else
            echo "$MATCHES content substitutions required: $FILENAME (dashes)"
            COUNT=$((COUNT+MATCHES))
        fi
        sed -i "s/$TEMPLATE_NAME/$REPO_NAME/g" "$FILENAME"
    fi
    if (grep "$ALT_TEMPLATE_NAME" "$FILENAME" > /dev/null 2>&1); then
        MATCHES=$(grep -c "$ALT_TEMPLATE_NAME" "$FILENAME")
        if [ "$MATCHES" -eq 1 ]; then
            echo "1 content substitution required: $FILENAME (underscores)"
            COUNT=$((COUNT++))
        else
            echo "$MATCHES content substitutions required: $FILENAME (underscores)"
            COUNT=$((COUNT+MATCHES))
        fi
        sed -i "s/$ALT_TEMPLATE_NAME/$ALT_REPO_NAME/g" "$FILENAME"
    fi
    if [[ "$COUNT" != "0" ]] && [[ "$COUNT" = "1" ]]; then
        echo "$COUNT substitution made in file: $FILENAME"
    elif [[ "$COUNT" != "0" ]] && [[ "$COUNT" -gt "1" ]]; then
        echo "$COUNT substitutions made in file: $FILENAME"
    fi
}

### Main script entry point

if ! (git rev-parse --show-toplevel > /dev/null); then
    echo "Error: this folder is not part of a GIT repository"; exit 1
fi

REPO_DIR=$(git rev-parse --show-toplevel)
REPO_NAME=$(basename "$REPO_DIR")
ALT_REPO_NAME="${REPO_NAME//-/_}"

if [ "$TEMPLATE_NAME" == "$REPO_NAME" ]; then
    echo "WARNING: template name matches repository name"
else
    echo "Template name: $TEMPLATE_NAME"
    echo "Alternate name: $ALT_TEMPLATE_NAME"
    echo "Repository name: $REPO_NAME"
    echo "Alternate name: $ALT_REPO_NAME"
fi

# Change to top-level of GIT repository
CURRENT_DIR=$(pwd)
if [ "$REPO_DIR" != "$CURRENT_DIR" ]; then
    echo "Changing directory to: $REPO_DIR"
    if ! (cd "$REPO_DIR"); then
        echo "Could not change directory!"; exit 1
    fi
fi

echo "Processing repository contents..."

# Rename directories first, as they affect file paths afterwards
for FS_OBJECT in $(find -- * -type d | xargs -0); do
    rename_object "$FS_OBJECT"
    if [ -f "$FS_OBJECT" ]; then
        file_content_substitution "$FS_OBJECT"
    fi
done

for FS_OBJECT in $(find -- * -type f | xargs -0); do
    rename_object "$FS_OBJECT"
    if [ -f "$FS_OBJECT" ]; then
        file_content_substitution "$FS_OBJECT"
    fi
done
