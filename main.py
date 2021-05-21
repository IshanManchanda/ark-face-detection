# TODO: Open webcam and get frames
# TODO: Detect face and get coordinates
# TODO: Create GUI
# TODO: Draw ball and move it around
import cv2
import numpy as np
import os

base_path = os.path.dirname(os.path.abspath(__file__))
face_cascade = cv2.CascadeClassifier(
	os.path.join(base_path, 'resources/haarcascade_frontalface.xml')
)


def find_face(img):
	# Convert the img to grayscale and then run haarcascade-based
	# face detection
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	# No faces found, return False for success
	if len(faces) == 0:
		return False, None

	# Multiple faces found, we'll pick the face with the largest bounding area
	if len(faces) > 1:
		return True, max(faces, key=lambda f: f[2] * f[3])

	# Found a single face, return it along with True for success
	return True, faces[0]


def main():
	cam = cv2.VideoCapture(0)
	cv2.namedWindow("test")

	while True:
		ret, frame = cam.read()
		if not ret:
			print("failed to grab frame")
			break
		print(find_face(frame))
		cv2.imshow("test", frame)
		k = cv2.waitKey(1)
	cam.release()


if __name__ == '__main__':
	main()
