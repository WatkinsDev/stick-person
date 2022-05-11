import cv2

ball_path = "./frames_output/img-0001_ball.png"
stick_path = "./frames_output/img-0001_stick.png"

img1 = cv2.imread(ball_path)
img2 = cv2.imread(stick_path)

one = img1.resize((220, 180))
two = img2.resize((220, 180))

dst = cv2.addWeighted(one, 0.5, two, 0.7, 0)

cv2.imshow("Blended Image", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
