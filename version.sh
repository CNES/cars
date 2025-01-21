#!/bin/sh

if [ "$1" = "get-vcs" ]; then
  python -m setuptools_scm
elif [ "$1" = "set-dist" ]; then
  echo $MESON_PROJECT_DIST_ROOT
  $MESONREWRITE --sourcedir="$MESON_PROJECT_DIST_ROOT" kwargs set project / version "$2"
else
  exit 1
fi
