import numpy as np
import cv2


class Ball:
	def __init__(self, pos, vel, radius, bounds):
		self.pos = pos
		self.vel = vel
		self.radius = radius
		self.bounds = bounds

	def draw(self, img):
		# Draw the ball on the given image
		cv2.circle(img, self.pos, self.radius, (255, 150, 50), -1)

	def update(self, head_pos, head_radius):
		# Update position
		self.pos += self.vel

		# Check for collision with head
		self.check_collision(head_pos, head_radius)

		# Check for collision with boundaries and return game_over flag
		return self.check_bounds()

	def check_bounds(self):
		# Check if ball is touching bottom edge
		if self.pos[1] + self.radius - self.vel[1] >= self.bounds[1]:
			# If yes, game over so return True
			return True

		# Check if ball is touching top edge, rebound if yes
		if self.pos[1] - self.radius <= 0:
			self.vel[1] *= -1

		# Check if ball is touching left or right edges, rebound if yes
		if self.pos[0] - self.radius <= 0:
			self.vel[0] *= -1
		elif self.pos[0] + self.radius >= self.bounds[0]:
			self.vel[0] *= -1

		# Game not over, return False
		return False

	def check_collision(self, head_pos, head_radius):
		# Get squared distance between head and ball
		dist_sq = np.sum(np.square(head_pos - self.pos))

		# If this is less than the square of the sum of the radii, collision
		if dist_sq <= (head_radius + self.radius) ** 2:
			# TODO: Improve collision logic

			# If ball is much higher, just rebound y
			if self.pos[1] + head_radius * 0.7 < head_pos[1]:
				self.vel[1] = -abs(self.vel[1])

			# If ball is near middle, rebound both
			elif self.pos[1] < head_pos[1]:
				self.vel[1] = -abs(self.vel[1])

				sign = 1 if self.pos[0] >= head_pos[0] else -1
				self.vel[0] = sign * abs(self.vel[0])

			# If ball is below, rebound x and let game end
			else:
				sign = 1 if self.pos[0] >= head_pos[0] else -1
				self.vel[0] = sign * abs(self.vel[0])
