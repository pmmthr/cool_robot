from random import shuffle
import math
import utils

def get_sensor_values(robot_x, robot_y, maze_walls):
    
    distances = []
    for angle_offset in angle_offsets:
        angle = math.radians(angle_offset)
        for dist in range(1, MAX_SENSOR_RANGE + 1):
            sensor_x = robot_x + dist * math.cos(angle)
            sensor_y = robot_y + dist * math.sin(angle)
            if any(wall.collidepoint(sensor_x, sensor_y) for wall in maze_walls):
                distances.append(dist)
                break
        else:
            distances.append(MAX_SENSOR_RANGE)  
    return distances

def get_observation(robot_x, robot_y, robot_angle, landmarks):
	
	# uncomment for debug - this method ia a simple simulation
	# of obvervation process
	# with perfect accuracy, returns robot_x, robot_y, robot_angle
	# return robot_x, robot_y, robot_angle

	# robot_x, robot_y are used to simulate if landmarks are in range
	close_landmarks = []
	for l in landmarks:
		l_x, l_y = l.center
		if get_dist(robot_x, robot_y, l_x, l_y) <= MAX_SENSOR_RANGE:
			close_landmarks.append(l)

	if len(close_landmarks) <= 2:
		return None
	shuffle(close_landmarks)
	
	dist = []
	c = []
	# robot_x, robot_y are used to simulate beam sensors and get distances to landmarks
	for l in close_landmarks:
		l_x, l_y = l.center
		c.append(l.center)
		dist.append(get_dist(robot_x, robot_y, l_x, l_y))

	# deduce x and y observations from landmark data and sensor measurements
	new_x, new_y = circle_intersection(c[0], dist[0], c[1], dist[1])

	# robot_angle is used to simulate measuring relative angle to landmarks
	angle = []
	for l in close_landmarks:
		l_x, l_y = l.center
		angle.append(math.degrees(math.atan2(l_y - new_y, l_x - new_x)) - robot_angle)
	angle = [a % 360 for a in angle]

	relative_angle = angle[0]
	al_x, al_y = close_landmarks[0].center
	landmark_angle = math.degrees(math.atan2(al_y - new_y, al_x - new_x))

	# deduce robot angle observation from landmark data and sensor measurements
	new_angle = landmark_angle - relative_angle
	new_angle = new_angle % 360

	return new_x, new_y, new_angle