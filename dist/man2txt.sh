#!/bin/bash

while IFS= read -r cmd; do
   mandoc -mdoc -c -T utf8 "$(man -w ${cmd})" | col -b > "$cmd".txt 2>error.err 
done < <(compgen -c | sort | uniq)
find . -type f -empty -delete
