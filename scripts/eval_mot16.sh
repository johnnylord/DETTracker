#!/bin/bash

if [[ $# != 2 ]]; then
    echo "Please enter grountruths and tests"
    exit 1
fi

NAME=$(basename $2)

rm -rf "/tmp/${NAME}"
cp -r $2 "/tmp/${NAME}"

find "/tmp/${NAME}" -type f -regex ".*\.txt" -exec mv '{}' "/tmp/${NAME}" \;
find "/tmp/${NAME}" -maxdepth 1 -mindepth 1 -type d -exec rm -rf '{}' \;

python3 -m motmetrics.apps.evaluateTracking $1 "/tmp/${NAME}" seqmaps/MOT16
