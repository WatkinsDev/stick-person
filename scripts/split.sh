#Prep
cd frames
rm *.png;

#Video to images
# ffmpeg -i ../input/initial_video.mp4 -vf fps=25 img-%04d.png;
ffmpeg -i ../input/initial_video.mp4 -vf fps=2 img-%04d.png;
