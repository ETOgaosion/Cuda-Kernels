#!/bin/bash

pid=$(ps aux | grep '\./occupy_gpu' | grep -v grep | awk '{print $2}' | head -n 1)

if [ -z "$pid" ]; then
    echo "not found 'project_pactum' process."
else
    dcgmi dmon -e 1002,1003,1004,1006,1007,1008,1005,1009,1001,1011,1012 > res.txt
fi