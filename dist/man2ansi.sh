#!/bin/bash

while IFS= read -r cmd; do
  pandoc --from man --to ansi "$(man -w ${cmd})" | col -b >"$cmd".txt 2>error.err
done < <(compgen -c | sort | uniq)
find . -type f -size 1c -delete
