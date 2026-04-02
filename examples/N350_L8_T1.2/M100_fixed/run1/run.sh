#!/bin/bash
cd "$(dirname "$0")"
julia main.jl > julia_out.txt 2>&1
echo "Exit code: $?" >> julia_out.txt
