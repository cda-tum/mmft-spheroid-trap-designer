#!/bin/sh
cd $(dirname "$0") || exit 1
blockMesh &&
surfaceFeatures &&
snappyHexMesh -overwrite &&
simpleFoam
