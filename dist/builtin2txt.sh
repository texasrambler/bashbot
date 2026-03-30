#!/bin/bash

while IFS= read -r cmd; do
  echo "$(help ${cmd})" >"$cmd".txt 2>error.err
done < <(compgen -b | sort | uniq)
