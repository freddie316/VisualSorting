@echo off
set methods=bubble cocktail selection insertion shell quick merge radix  

echo Available methods:
(for %%i in (%methods%) do (
	echo %%i
))

set /p sort=Select method: 

set /p N=List length: 

python Sorts.py %sort% %N%

ffmpeg -y -r 60 -i "frames/%sort%_frame%%05d.png" -i "sound/%sort%_sound.wav" -c:v libx264 -preset veryslow -crf 0 -map 0:v -map 1:a "movie/%sort%Sort.mp4"

del /S /Q frames\*.png

start movie/%sort%Sort.mp4