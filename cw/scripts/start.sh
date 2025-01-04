#!/bin/bash

set -e  # Прерывание скрипта при любой ошибке

rm -rf output img
mkdir output img

echo "Running solution"
if ! ./"$1" "$3" < default.in; then
    echo "Solution failed. Exiting."
    exit 1
fi

echo "Running conv.py"
for (( k = 0; k < $2; k++ )); do
    bin="./output/$k.data"
    if [ ! -f "$bin" ]; then
        break
    fi
    img="./img/$(printf "%03d" $((k+1))).png"
    python3 conv.py "$bin" "$img"
done

echo "Running ffmpeg"
ffmpeg -framerate 15 -i ./img/%03d.png -pix_fmt yuv420p vid.mp4
# ffmpeg -framerate 15 -i ./img/%03d.png -vf "scale=iw:-1:flags=lanczos" -y vid.gif
