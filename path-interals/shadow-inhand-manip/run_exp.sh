#!/bin/bash

for i in {1..5}
do
    echo "Running trial no $i"
    ./known_uncertainty_percent_succ.py $i
done
