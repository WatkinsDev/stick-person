# Images to video
cd ./frames_output
rm *.mp4; 
ffmpeg -framerate 2 -i img-%*.png -s:v 1280x720 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p stick_person.mp4; 
# ffmpeg -framerate 25 -i img-%*.png -s:v 1280x720 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p stick_person.mp4; 

# Strip Audio
# cd $1
# ffmpeg -i $2 -map 0:a audio.wav -map 0:v ignore.avi 

# Audio added to final file
# ffmpeg -i $2_cartoon.mp4 -i audio.wav -c:v copy -c:a aac $2_cartoon_with_audio.mp4

# Play
vlc stick_person.mp4;