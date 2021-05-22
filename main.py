import cv2
import numpy as np
import os

from ball import Ball

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
		return False, (0, 0, 0, 0)

	# Multiple faces found, we'll pick the face with the largest bounding area
	if len(faces) > 1:
		return True, max(faces, key=lambda f: f[2] * f[3])

	# Found a single face, return it along with True for success
	return True, faces[0]


def main():
	# cam = cv2.VideoCapture(0)
	cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
	cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

	size = np.array([1280, 720])
	cv2.namedWindow("game", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("game", *size)

	pos = np.array([100, 100])
	vel = 15 * np.array([2, 2])
	ball = Ball(pos, vel, 60, size)

	head_pos = np.array([0, 0])
	head_radius = 0

	while True:
		ret, frame = cam.read()
		if not ret:
			print("failed to grab frame")
			break
		frame = cv2.flip(frame, 1)

		success, (x, y, w, h) = find_face(frame)
		if success:
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			head_pos = np.array([x + w // 2, y + h // 2])
			head_radius = max(w // 2, h // 2)
			cv2.circle(frame, head_pos, head_radius, (50, 50, 255), 2)

		game_over = ball.update(head_pos, head_radius)
		if game_over:
			print("Game Over.")
			break
		ball.draw(frame)

		cv2.imshow("game", frame)
		k = cv2.waitKey(1)
		# Check if k == ESC and break

	cam.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
