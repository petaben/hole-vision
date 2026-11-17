# instructions for claude

Create a python script that can measure the size of holes in pictures that were taken with a microscope mounted camera.
You can find example pictures in the /test_data folder

The script takes `path to image` and `px/mm config (float)` as input and returns `hole diameter in mm`

When passing `-d` (debug), the script will as well generate an image.
The generated image is the input image + a circle marking the detected hole