import numpy
import math


def isZero (v):
	minV = 1e-5
	if v > -minV and v < minV:
		return True
	return False

def length_vec3 (v):
	vec3 = numpy.array(v)
	return numpy.linalg.norm(vec3)

def normalize_vec3 (v):
	vec3 = numpy.array(v)
	len = numpy.linalg.norm(vec3)
	if isZero(len):
		return [0, 0, 0]
	return vec3 / len

def normalize_vec4 (v):
	vec4 = numpy.array(v)
	len = numpy.linalg.norm(vec4)
	if isZero(len):
		return [0, 0, 0, 0]
	return vec4 / len

def dot_vec3 (v1, v2):
	vec3_1 = numpy.array(v1)
	vec3_2 = numpy.array(v2)
	return numpy.dot(vec3_1, vec3_2)


def cross_vec3 (v1, v2):
	vec3_1 = numpy.array(v1)
	vec3_2 = numpy.array(v2)
	return numpy.cross(vec3_1, vec3_2)


def quaternion_a_b (a, b):
	a = normalize_vec3(a)
	b = normalize_vec3(b)
	q = [0.0, 0.0, 0.0, 0.0]

	c = cross_vec3(b, a)
	d = -length_vec3(c)
	c = normalize_vec3(c)

	epsilon = 0.0002
	ip = dot_vec3(a, b)
	if -epsilon < d or 1.0 < ip:
		if ip < (epsilon - 1.0):
			a2 = [-a[1], a[2], a[0]]
			c = normalize_vec3(cross_vec3(a2, a))
			q[0] = 0.0
			q[1] = c[0]
			q[2] = c[1]
			q[3] = c[2]
		else:
			q = numpy.array([1.0, 0.0, 0.0, 0.0])
	else:
		e = c * math.sqrt(0.5 * (1.0 - ip))
		q[0] = math.sqrt(0.5 * (1.0 + ip))
		q[1] = e[0]
		q[2] = e[1]
		q[3] = e[2]
	return q


def quaternion_matrix4x4 (q):
	m = numpy.matrix(numpy.identity(4))


	q2 = range(4)
	q2[0] = q[1]
	q2[1] = q[2]
	q2[2] = q[3]
	q2[3] = q[0]

	m[0,0] = q2[3]*q2[3] + q2[0]*q2[0] - q2[1]*q2[1] - q2[2]*q2[2]
	m[0,1] = 2.0 * q2[0] * q2[1] - 2.0 * q2[3] * q2[2]
	m[0,2] = 2.0 * q2[0] * q2[2] + 2.0 * q2[3] * q2[1]
	m[0,3] = 0.0

	m[1,0] = 2.0 * q2[0] * q2[1] + 2.0 * q2[3] * q2[2]
	m[1,1] = q2[3] * q2[3] - q2[0] * q2[0] + q2[1] * q2[1] - q2[2] * q2[2]
	m[1,2] = 2.0 * q2[1] * q2[2] - 2.0 * q2[3] * q2[0]
	m[1,3] = 0.0

	m[2,0] = 2.0 * q2[0] * q2[2] - 2.0 * q2[3] * q2[1]
	m[2,1] = 2.0 * q2[1] * q2[2] + 2.0 * q2[3] * q2[0]
	m[2,2] = q2[3] * q2[3] - q2[0] * q2[0] - q2[1] * q2[1] + q2[2] * q2[2]
	m[2,3] = 0.0

	m[3,0] = 0.0
	m[3,1] = 0.0
	m[3,2] = 0.0
	m[3,3] = q2[3] * q2[3] + q2[0] * q2[0] + q2[1] * q2[1] + q2[2] * q2[2]

	k = m[3,3]
	for i in range(3):
		for j in range(3):
			m[i,j] /= k

	m[3,3] = 1.0

	return m


def rotate_a_to_b_matrix4x4 (_a, _b):
	#m = numpy.matrix(numpy.identity(4))
	a = numpy.array([_a[0], _a[1], _a[2]])
	b = numpy.array([_b[0], _b[1], _b[2]])
	return quaternion_matrix4x4( quaternion_a_b(a, b) )


if __name__ == "__main__":
    #a = [0, 0.2, 1.0]
    #b = [1.0, 0.2, 0.0]
    #a = [0.07834, -0.9969, -0.00650]
    #b = [0.99070, 0.07857, -0.111052]
    #a = [0.529017, -0,801472, 0.278896]
    #b = [0.84852, 0.504181, -0.160629]
    #a = [0.978708, -0.197813, -0.0547758]
    #b = [0.182067, 0.959871, -0.213306]
    a = [-0.118472, 0.0317288, -0.99245]
    b = [-0.91429, 0.386603, 0.121492]


    m = rotate_a_to_b_matrix4x4(a, b)
    print (m)