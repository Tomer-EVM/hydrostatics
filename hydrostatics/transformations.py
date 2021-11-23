import numpy as np


def get_closest_divisors(n: int):
    """The closest two divisors of an integer

    Parameters
    ----------
    n : int
        integer to find divisors of.
        Must not be zero.

    Returns
    -------
    int, int
        Two integer divisors of n

    Notes
    -----
    Useful for automatically detecting the shape of the UV when we know it's approximately square
    """
    x = round(np.sqrt(n))
    while n % x != 0:
        x -= 1
    return (int(x), int(n // x))


def transformation_matrix(x: float, y: float, z: float):
    """Creates a transformation matrix for rotational values

    Rotations must be given in degrees
    Currently using the method described by https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations

    Parameters
    ----------
    x : float
    y : float
    z : float

    Returns
    -------
    np.array
        3x3 rotation matrix
    """
    z = np.deg2rad(z)
    y = np.deg2rad(y)
    x = np.deg2rad(x)

    R_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(x), -np.sin(x)], [0.0, np.sin(x), np.cos(x)]]
    )
    R_y = np.array(
        [[np.cos(y), 0.0, np.sin(y)], [0.0, 1.0, 0.0], [-np.sin(y), 0.0, np.cos(y)]]
    )
    R_z = np.array(
        [[np.cos(z), -np.sin(z), 0.0], [np.sin(z), np.cos(z), 0.0], [0.0, 0.0, 1.0]]
    )
    return R_x @ R_y @ R_z


'''
def clip_points(a, b, c, normal=np.array([0.,0.,1.]), position=np.array([0.,0.,0.])):
    """ Clips a face described by three points to below 0

    Parameters
    ----------
    a, b, c : array-like
        Three points that descibe a face in 3D
    normal : array-like, optional
        Unit vector in the direction of the normal of the plane (default is [0,0,1])
    position : array-like, optional
        A point on the plane (default is [0,0,0])

    Returns
    -------
    t : list
        list of Tri objects. There will usually be one. If it is clipped, can have 2. 
        Can also have 0 if completely above.
    e : list
        list of Edge objects. This is for clipped edges. 
        Potentially one, if clipped, otherwise an empty list

    Notes
    -----
    See http://paulbourke.net/geometry/polygonmesh/source3.c for the original
    """

    a_s = a.dot(normal) - normal.dot(position)
    b_s = b.dot(normal) - normal.dot(position)
    c_s = c.dot(normal) - normal.dot(position)

    t = []
    e = []

    if a_s >= 0 and b_s >= 0 and c_s >= 0:
        #return [], []
        pass
    elif a_s > 0 and b_s < 0 and c_s < 0:
        d = a - a_s * (c - a) / (c_s - a_s)
        a = a - a_s * (b - a) / (b_s - a_s)
        #return [Tri(a,b,c), Tri(a,c,d)], [Edge(a,d)]
        t.append(Tri(a,b,c))
        t.append(Tri(a,c,d))
        e.append(Edge(a,d))
    elif b_s > 0 and a_s < 0 and c_s < 0:
        d = c
        c = b - b_s * (c - b) / (c_s - b_s)
        b = b - b_s * (a - b) / (a_s - b_s)
        t.append(Tri(a,b,c))
        t.append(Tri(a,c,d))
        e.append(Edge(c,b))
        #return [Tri(a,b,c), Tri(a,c,d)], [Edge(c,b)]
    elif c_s > 0 and a_s < 0 and b_s < 0:
        d = c - c_s * (a - c) / (a_s - c_s)
        c = c - c_s * (b - c) / (b_s - c_s)
        t.append(Tri(a,b,c))
        t.append(Tri(a,c,d))
        e.append(Edge(d,c))
        #return [Tri(a,b,c), Tri(a,c,d)], [Edge(d,c)]
    elif a_s < 0 and b_s >= 0 and c_s >= 0:
        b = a - a_s * (b - a) / (b_s - a_s)
        c = a - a_s * (c - a) / (c_s - a_s)
        t.append(Tri(a,b,c))
        e.append(Edge(c,b))
        #return [Tri(a,b,c)], [Edge(c,b)]
    elif b_s < 0 and a_s >= 0 and c_s >= 0:
        a = b - b_s * (a - b) / (a_s - b_s)
        c = b - b_s * (c - b) / (c_s - b_s)
        t.append(Tri(a,b,c))
        e.append(Edge(a,c))
        #return [Tri(a,b,c)], [Edge(a,c)]
    elif c_s < 0 and a_s >= 0 and b_s >= 0:
        a = c - c_s * (a - c) / (a_s - c_s)
        b = c - c_s * (b - c) / (b_s - c_s)
        t.append(Tri(a,b,c))
        e.append(Edge(b,a))
        #return [Tri(a,b,c)], [Edge(b,a)]
    else:
        t.append(Tri(a,b,c))
    #return [Tri(a,b,c)], []
    return t, e
'''

if __name__ == "__main__":
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])

    angles = (0, 0, 0)
    print(angles)
    R = transformation_matrix(*angles)
    print(R @ x)
    print(R @ y)
    print(R @ z)

    angles = (90, 0, 0)
    print(angles)
    R = transformation_matrix(*angles)
    print(R @ x)
    print(R @ y)
    print(R @ z)

    angles = (0, 90, 0)
    print(angles)
    R = transformation_matrix(*angles)
    print(R @ x)
    print(R @ y)
    print(R @ z)

    angles = (0, 0, 90)
    print(angles)
    R = transformation_matrix(*angles)
    print(R @ x)
    print(R @ y)
    print(R @ z)

    angles = (90, 90, 0)
    print(angles)
    R = transformation_matrix(*angles)
    print(R @ x)
    print(R @ y)
    print(R @ z)

    angles = (90, 0, 90)
    print(angles)
    R = transformation_matrix(*angles)
    print(R @ x)
    print(R @ y)
    print(R @ z)

    angles = (0, 90, 90)
    print(angles)
    R = transformation_matrix(*angles)
    print(R @ x)
    print(R @ y)
    print(R @ z)

    angles = (90, 90, 90)
    print(angles)
    R = transformation_matrix(*angles)
    print(R @ x)
    print(R @ y)
    print(R @ z)
