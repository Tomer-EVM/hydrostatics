cimport cython
import numpy as np
cimport numpy as cnp

@cython.boundscheck(False)
@cython.wraparound(False)
def clip_points(cython.floating[:,::1] vertices, cython.integral[:,::1] faces, cython.floating[:] position, cython.floating[:] normal):
    
    cdef long i
    cdef long j
    cdef Py_ssize_t N = faces.shape[0]
    cdef Py_ssize_t M = vertices.shape[0]

    cdef double p_n = dot3(position, normal)
    cdef double[:] v_n = np.empty(M, np.float64)
    cdef double[:] v_p_n = np.empty(M, np.float64)

    v_n = vertices.base @ normal.base
    v_p_n = v_n.base - p_n
    #for i in range(M):
        #v_n[i] = dot3(vertices[i,:], normal)
        #v_p_n[i] = v_n[i] - p_n

    cdef long n_one_below = 0
    cdef long n_two_below = 0
    cdef long n_three_below = 0
    cdef long[:] n_below = np.empty(N, np.int_)
    
    cdef long c
    for i in range(N):
        c = (v_p_n[faces[i,0]]<0) + (v_p_n[faces[i,1]]<0) + (v_p_n[faces[i,2]]<0)
        n_below[i] = c
        if c == 1: n_one_below += 1
        elif c == 2: n_two_below += 1
        elif c == 3: n_three_below += 1
    
    face_out = np.empty((n_one_below+2*n_two_below+n_three_below, 3), np.int_)
    edge_out = np.empty((n_one_below+n_two_below, 2), np.int_)
    vertex_out = np.vstack((np.copy(vertices), np.empty((2*n_one_below+2*n_two_below, 3), np.float64)))
    cdef long[:,::1] f = face_out 
    cdef long[:,::1] e = edge_out
    cdef double[:,::1] v = vertex_out

    cdef long vertex_c = M
    cdef long face_c = 0
    cdef long edge_c = 0
    for i in range(N):
        if n_below[i] == 3:
            for j in range(3):
                f[face_c,j] = faces[i,j]
            face_c += 1
        elif n_below[i] == 2:
            if v_p_n[faces[i,0]]>=0:

                f[face_c,0] = vertex_c
                f[face_c,1] = faces[i,1]
                f[face_c,2] = faces[i,2]

                f[face_c+1,0] = vertex_c+1
                f[face_c+1,1] = vertex_c
                f[face_c+1,2] = faces[i,2]

                e[edge_c,0] = vertex_c
                e[edge_c,1] = vertex_c+1

                face_c += 2
                edge_c += 1

                v[vertex_c,0] = vertices[faces[i,0],0] - v_p_n[faces[i,0]] * (vertices[faces[i,1],0] - vertices[faces[i,0],0]) / (v_n[faces[i,1]]-v_n[faces[i,0]])
                v[vertex_c,1] = vertices[faces[i,0],1] - v_p_n[faces[i,0]] * (vertices[faces[i,1],1] - vertices[faces[i,0],1]) / (v_n[faces[i,1]]-v_n[faces[i,0]])
                v[vertex_c,2] = vertices[faces[i,0],2] - v_p_n[faces[i,0]] * (vertices[faces[i,1],2] - vertices[faces[i,0],2]) / (v_n[faces[i,1]]-v_n[faces[i,0]])
                vertex_c += 1
                
                v[vertex_c,0] = vertices[faces[i,0],0] - v_p_n[faces[i,0]] * (vertices[faces[i,2],0] - vertices[faces[i,0],0]) / (v_n[faces[i,2]]-v_n[faces[i,0]])
                v[vertex_c,1] = vertices[faces[i,0],1] - v_p_n[faces[i,0]] * (vertices[faces[i,2],1] - vertices[faces[i,0],1]) / (v_n[faces[i,2]]-v_n[faces[i,0]])
                v[vertex_c,2] = vertices[faces[i,0],2] - v_p_n[faces[i,0]] * (vertices[faces[i,2],2] - vertices[faces[i,0],2]) / (v_n[faces[i,2]]-v_n[faces[i,0]])
                vertex_c += 1

            elif v_p_n[faces[i,1]]>=0:
                f[face_c,0] = faces[i,0]
                f[face_c,1] = vertex_c
                f[face_c,2] = faces[i,2]

                f[face_c+1,0] = vertex_c
                f[face_c+1,1] = vertex_c+1
                f[face_c+1,2] = faces[i,2]

                e[edge_c,0] = vertex_c+1
                e[edge_c,1] = vertex_c

                face_c += 2
                edge_c += 1

                v[vertex_c,0] = vertices[faces[i,1],0] - v_p_n[faces[i,1]] * (vertices[faces[i,0],0] - vertices[faces[i,1],0]) / (v_n[faces[i,0]]-v_n[faces[i,1]])
                v[vertex_c,1] = vertices[faces[i,1],1] - v_p_n[faces[i,1]] * (vertices[faces[i,0],1] - vertices[faces[i,1],1]) / (v_n[faces[i,0]]-v_n[faces[i,1]])
                v[vertex_c,2] = vertices[faces[i,1],2] - v_p_n[faces[i,1]] * (vertices[faces[i,0],2] - vertices[faces[i,1],2]) / (v_n[faces[i,0]]-v_n[faces[i,1]])
                vertex_c += 1
                
                v[vertex_c,0] = vertices[faces[i,1],0] - v_p_n[faces[i,1]] * (vertices[faces[i,2],0] - vertices[faces[i,1],0]) / (v_n[faces[i,2]]-v_n[faces[i,1]])
                v[vertex_c,1] = vertices[faces[i,1],1] - v_p_n[faces[i,1]] * (vertices[faces[i,2],1] - vertices[faces[i,1],1]) / (v_n[faces[i,2]]-v_n[faces[i,1]])
                v[vertex_c,2] = vertices[faces[i,1],2] - v_p_n[faces[i,1]] * (vertices[faces[i,2],2] - vertices[faces[i,1],2]) / (v_n[faces[i,2]]-v_n[faces[i,1]])
                vertex_c += 1

            elif v_p_n[faces[i,2]]>=0:
                f[face_c,0] = faces[i,0]
                f[face_c,1] = faces[i,1]
                f[face_c,2] = vertex_c

                f[face_c+1,0] = vertex_c
                f[face_c+1,1] = vertex_c+1
                f[face_c+1,2] = faces[i,0]

                e[edge_c,0] = vertex_c+1
                e[edge_c,1] = vertex_c

                face_c += 2
                edge_c += 1

                v[vertex_c,0] = vertices[faces[i,2],0] - v_p_n[faces[i,2]] * (vertices[faces[i,1],0] - vertices[faces[i,2],0]) / (v_n[faces[i,1]]-v_n[faces[i,2]])
                v[vertex_c,1] = vertices[faces[i,2],1] - v_p_n[faces[i,2]] * (vertices[faces[i,1],1] - vertices[faces[i,2],1]) / (v_n[faces[i,1]]-v_n[faces[i,2]])
                v[vertex_c,2] = vertices[faces[i,2],2] - v_p_n[faces[i,2]] * (vertices[faces[i,1],2] - vertices[faces[i,2],2]) / (v_n[faces[i,1]]-v_n[faces[i,2]])
                vertex_c += 1
                
                v[vertex_c,0] = vertices[faces[i,2],0] - v_p_n[faces[i,2]] * (vertices[faces[i,0],0] - vertices[faces[i,2],0]) / (v_n[faces[i,0]]-v_n[faces[i,2]])
                v[vertex_c,1] = vertices[faces[i,2],1] - v_p_n[faces[i,2]] * (vertices[faces[i,0],1] - vertices[faces[i,2],1]) / (v_n[faces[i,0]]-v_n[faces[i,2]])
                v[vertex_c,2] = vertices[faces[i,2],2] - v_p_n[faces[i,2]] * (vertices[faces[i,0],2] - vertices[faces[i,2],2]) / (v_n[faces[i,0]]-v_n[faces[i,2]])
                vertex_c += 1

        elif n_below[i] == 1:
            if v_p_n[faces[i,0]]<0:
                f[face_c,0] = faces[i,0]
                f[face_c,1] = vertex_c
                e[edge_c,1] = vertex_c
                v[vertex_c,0] = vertices[faces[i,0],0] - v_p_n[faces[i,0]] * (vertices[faces[i,1],0] - vertices[faces[i,0],0]) / (v_n[faces[i,1]]-v_n[faces[i,0]])
                v[vertex_c,1] = vertices[faces[i,0],1] - v_p_n[faces[i,0]] * (vertices[faces[i,1],1] - vertices[faces[i,0],1]) / (v_n[faces[i,1]]-v_n[faces[i,0]])
                v[vertex_c,2] = vertices[faces[i,0],2] - v_p_n[faces[i,0]] * (vertices[faces[i,1],2] - vertices[faces[i,0],2]) / (v_n[faces[i,1]]-v_n[faces[i,0]])
                vertex_c += 1
                
                f[face_c,2] = vertex_c
                e[edge_c,0] = vertex_c
                v[vertex_c,0] = vertices[faces[i,0],0] - v_p_n[faces[i,0]] * (vertices[faces[i,2],0] - vertices[faces[i,0],0]) / (v_n[faces[i,2]]-v_n[faces[i,0]])
                v[vertex_c,1] = vertices[faces[i,0],1] - v_p_n[faces[i,0]] * (vertices[faces[i,2],1] - vertices[faces[i,0],1]) / (v_n[faces[i,2]]-v_n[faces[i,0]])
                v[vertex_c,2] = vertices[faces[i,0],2] - v_p_n[faces[i,0]] * (vertices[faces[i,2],2] - vertices[faces[i,0],2]) / (v_n[faces[i,2]]-v_n[faces[i,0]])
                vertex_c += 1
                face_c += 1
                edge_c += 1

            elif v_p_n[faces[i,1]]<0:
                f[face_c,1] = faces[i,1]
                f[face_c,0] = vertex_c
                e[edge_c,0] = vertex_c
                v[vertex_c,0] = vertices[faces[i,1],0] - v_p_n[faces[i,1]] * (vertices[faces[i,0],0] - vertices[faces[i,1],0]) / (v_n[faces[i,0]]-v_n[faces[i,1]])
                v[vertex_c,1] = vertices[faces[i,1],1] - v_p_n[faces[i,1]] * (vertices[faces[i,0],1] - vertices[faces[i,1],1]) / (v_n[faces[i,0]]-v_n[faces[i,1]])
                v[vertex_c,2] = vertices[faces[i,1],2] - v_p_n[faces[i,1]] * (vertices[faces[i,0],2] - vertices[faces[i,1],2]) / (v_n[faces[i,0]]-v_n[faces[i,1]])
                vertex_c += 1
                
                f[face_c,2] = vertex_c
                e[edge_c,1] = vertex_c
                v[vertex_c,0] = vertices[faces[i,1],0] - v_p_n[faces[i,1]] * (vertices[faces[i,2],0] - vertices[faces[i,1],0]) / (v_n[faces[i,2]]-v_n[faces[i,1]])
                v[vertex_c,1] = vertices[faces[i,1],1] - v_p_n[faces[i,1]] * (vertices[faces[i,2],1] - vertices[faces[i,1],1]) / (v_n[faces[i,2]]-v_n[faces[i,1]])
                v[vertex_c,2] = vertices[faces[i,1],2] - v_p_n[faces[i,1]] * (vertices[faces[i,2],2] - vertices[faces[i,1],2]) / (v_n[faces[i,2]]-v_n[faces[i,1]])
                vertex_c += 1
                face_c += 1
                edge_c += 1

            elif v_p_n[faces[i,2]]<0:
                f[face_c,2] = faces[i,2]
                f[face_c,0] = vertex_c
                e[edge_c,1] = vertex_c
                v[vertex_c,0] = vertices[faces[i,2],0] - v_p_n[faces[i,2]] * (vertices[faces[i,0],0] - vertices[faces[i,2],0]) / (v_n[faces[i,0]]-v_n[faces[i,2]])
                v[vertex_c,1] = vertices[faces[i,2],1] - v_p_n[faces[i,2]] * (vertices[faces[i,0],1] - vertices[faces[i,2],1]) / (v_n[faces[i,0]]-v_n[faces[i,2]])
                v[vertex_c,2] = vertices[faces[i,2],2] - v_p_n[faces[i,2]] * (vertices[faces[i,0],2] - vertices[faces[i,2],2]) / (v_n[faces[i,0]]-v_n[faces[i,2]])
                vertex_c += 1
                
                f[face_c,1] = vertex_c
                e[edge_c,0] = vertex_c
                v[vertex_c,0] = vertices[faces[i,2],0] - v_p_n[faces[i,2]] * (vertices[faces[i,1],0] - vertices[faces[i,2],0]) / (v_n[faces[i,1]]-v_n[faces[i,2]])
                v[vertex_c,1] = vertices[faces[i,2],1] - v_p_n[faces[i,2]] * (vertices[faces[i,1],1] - vertices[faces[i,2],1]) / (v_n[faces[i,1]]-v_n[faces[i,2]])
                v[vertex_c,2] = vertices[faces[i,2],2] - v_p_n[faces[i,2]] * (vertices[faces[i,1],2] - vertices[faces[i,2],2]) / (v_n[faces[i,1]]-v_n[faces[i,2]])
                vertex_c += 1
                face_c += 1
                edge_c += 1


    return vertex_out, edge_out, face_out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double dot3(cython.floating[:] a, cython.floating[:] b):
    cdef double out
    out = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    return out