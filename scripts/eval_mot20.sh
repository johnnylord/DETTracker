#!/bin/bash

if [[ $# != 2 ]]; then
    echo "Please enter grountruths and tests"
    exit 1
fi

NAME=$(basename $2)

rm -rf /tmp/${NAME}
cp -r $2 /tmp/${NAME}

mv /tmp/${NAME}/**/*.txt /tmp/${NAME}
find /tmp/${NAME} -maxdepth 1 -mindepth 1 -type d -exec rm rf '{}' \;

python -m motmetrics.apps.evaludateTracking $1 $2 seqmaps/MOT20
