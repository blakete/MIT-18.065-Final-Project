#!/bin/bash

INPUT="$1"
OUTPUT="$2"
STEP="${3:-5}"

TOTAL=$(identify -format "%n" "$INPUT" | head -n 1)
TO_DELETE=$(seq $((STEP - 1)) $STEP $((TOTAL - 1)) | paste -sd, -)

convert "$INPUT" -coalesce -delete "$TO_DELETE" "$OUTPUT"

