#!/bin/bash

START=0

for (( i=$START; i<=$1; i++ ))
do
    let "j = ${i} + 1"
    # probably no need to go through gif
    webm -i ./pdftoimage/gif_generated/${i}-${j}.gif -foi='-r 50' ./pdftoimage/video_generated/${i}-${j}.mp4
    webm -i ./pdftoimage/gif_generated/${j}-${i}.gif -foi='-r 50' ./pdftoimage/video_generated/${j}-${i}.mp4

done

    #ffmpeg -i ./pdftoimage/video_generated/${i}-${j}.mp4 -vf reverse ./pdftoimage/video_generated/${j}-${i}.mp4
