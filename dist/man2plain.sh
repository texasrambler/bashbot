#!/bin/bash
while IFS= read -r cmd; do
   man -E utf8 "$cmd" 2>/dev/null | col -b | pandoc --from man --to plain >"$cmd".txt 2>error.err
done < <(compgen -c | sort | uniq)
find . -type f -size 1c -delete
