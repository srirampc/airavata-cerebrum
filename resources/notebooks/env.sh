#!/bin/bash

print_usage () {
  echo "USAGE:: $0 -n ENV_NAME [-d create/remove]"
  exit 0
}

CMD=create
while getopts "n:d:h" opt; do
  case $opt in
    d) CMD=$OPTARG ;;
    n) ENV_NAME=$OPTARG  ;;
    h) print_usage ;;
    *) echo "Invalid argument"; exit 1;;
  esac
done

if [ -z "$CMD" ]; then
  CMD=create
fi

case $CMD in
    create|remove) echo Running command "$CMD";;
    *)             echo Invalid "$CMD" argument; exit 1;;
esac

if [ "$CMD" == "create" ]; then
    conda env create -n "$ENV_NAME" -f environment.yml
else
    conda remove -n "$ENV_NAME" --all
fi
