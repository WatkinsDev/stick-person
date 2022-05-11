import os
from image_combine import image_combine
from image_to_ball import image_to_ball
from image_to_stick import image_to_stick

dir_path = r"./frames"
count = 0
for path in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path, path)):
        count += 1
        file_number = str(count).zfill(4)

        print(f"Running process for file #{file_number}")
        image_to_ball(file_number)
        image_to_stick(file_number)
        image_combine(file_number)