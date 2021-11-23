cimport cython
import numpy as np
from cython.parallel import prange



@cython.boundscheck(False)
@cython.wraparound(False)
def volume_properties(cython.floating[:,::1] vertices, cython.integral[:,::1] faces, cython.floating[:] centre):
    cdef long i
    cdef long j
    cdef Py_ssize_t N = faces.shape[0]
    cdef double v

    cdef double v_total = 0.0
    centroid = np.array([0.,0.,0.], dtype=float)
    cdef double[:] c = centroid
    
    for i in range(N):
        v = tetrahedron_volume(vertices, faces[i,:], centre)
        v_total += v
        for j in range(3):
            c[j] += v * (vertices[faces[i,0],j]+vertices[faces[i,1],j]+vertices[faces[i,2],j]+centre[j])/4.0
    
    if v_total > 0.0:
        for j in range(3):
            c[j] /= v_total
    else:
        for j in range(3):
            c[j] = 0.0
    
    return v_total, centroid
    
@cython.boundscheck(False)
@cython.wraparound(False)
def tetrahedron_volume(cython.floating[:,::1] vertices, cython.integral[:] face, cython.floating[:] centre):

    cdef double out

    cdef double v210
    cdef double v120
    cdef double v201
    cdef double v021
    cdef double v102
    cdef double v012

    v210 = (vertices[face[2],0]-centre[0]) * (vertices[face[1],1]-centre[1]) * (vertices[face[0],2]-centre[2])
    v120 = (vertices[face[1],0]-centre[0]) * (vertices[face[2],1]-centre[1]) * (vertices[face[0],2]-centre[2])
    v201 = (vertices[face[2],0]-centre[0]) * (vertices[face[0],1]-centre[1]) * (vertices[face[1],2]-centre[2])
    v021 = (vertices[face[0],0]-centre[0]) * (vertices[face[2],1]-centre[1]) * (vertices[face[1],2]-centre[2])
    v102 = (vertices[face[1],0]-centre[0]) * (vertices[face[0],1]-centre[1]) * (vertices[face[2],2]-centre[2])
    v012 = (vertices[face[0],0]-centre[0]) * (vertices[face[1],1]-centre[1]) * (vertices[face[2],2]-centre[2])

    out = (-v210 + v120 + v201 - v021 - v102 + v012)/6.0

    return out