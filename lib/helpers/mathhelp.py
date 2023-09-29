"""
piperator - plugin for generating pipes in blender
Copyright (C) 2019  Thomas Meschede

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import math
from collections import defaultdict
#import sympy as sp

# comment out if no symbolic computations are allowed:
#cos = sp.cos
#sin = sp.sin
#atan2 = sp.atan2
#asin = sp.asin
from math import sin, cos, atan2, asin

pi = math.pi
pi2 = pi * 2.0
pi05 = pi * 0.5
deg = pi / 180.0
rad = 180.0 / pi


def vec(x): return np.array(x)


def vec3(x, y, z): return np.array((x, y, z))


def norm(x): return np.sqrt(x.dot(x))

def normalized(x): return x * (1.0 / norm(x))

#def vectorized_norm(x): return np.sqrt(np.sum(x*x, axis=1))
def vectorized_norm(x): return np.array([normalized(v) for v in x])


def vec_length(x): return np.sqrt(sum(x * x))
epsilon = 0.0001  # minimum accuracy for iterations


def ortho(x):
    """generate arbitrary orthogonal vector"""
    if x[0] == 0:
        return np.array([1, 0, 0])
    if x[1] == 0:
        return np.array([0, 1, 0])
    if x[2] == 0:
        return np.array([0, 0, 1])
    
    idx = max(range(len(x)), key=x.__getitem__)
    
    if idx == 0:    
        y = -(x[1] + x[2]) / x[0]
        return normalized(vec((y,1.0,1.0)))
    elif idx == 1:  
        y = -(x[0] + x[2]) / x[1]
        return normalized(vec((1.0,y,1.0)))
    elif idx == 2:
        y = -(x[0] + x[1]) / x[2]
        return normalized(vec((1.0,1.0,y)))

def eulerzyxmat(phi, theta, psi):
    sp, cp = sin(phi), cos(phi)
    sth, cth = sin(theta), cos(theta)
    sps, cps = sin(psi), cos(psi)
    rotmat = np.array([[cps * cth, -cp * sps + cps * sp * sth,
                        cp * cps * sth + sp * sps],
                       [cth * sps, cp * cps + sp * sps * sth,
                        cp * sps * sth - cps * sp],
                       [-sth, cth * sp, cp * cth]])
    return rotmat

def calc_directed_angle(u,v,n):
    """calculates a directed angle where
    the rotation axis is in the same direction as the normal vector n
    meaning: (u x v)*n > 0.
    """
    cr = np.cross(u,v)
    
    u_n = norm(u)
    v_n = norm(v)
    
    sgn = (cr @ n > 0) * 2 - 1
    
    res = np.sum(u*v) / (u_n * v_n)
    t = np.clip(res,-1.0,1.0)
    angle = sgn * np.arccos(t)
    return angle


def calc_angle_vec(u, v):
    """
    >>> u = vec((1.0,1.0,0.0))
    >>> v = vec((1.0,0.0,0.0))
    >>> calc_angle_vec(u,v)*rad
    45.00000000000001
    >>> u = vec((1.0,0.0,0.0))
    >>> v = vec((-1.0,0.0,0.0))
    >>> calc_angle_vec(u,v)*rad
    180.0
    >>> u = vec([-9.38963669e-01, 3.44016319e-01, 1.38777878e-17])
    >>> v = vec([-0.93896367, 0.34401632, 0.])
    >>> u @ v / (norm(v)*norm(u))
    1.0000000000000002
    >>> calc_angle_vec(u,v)*rad
    0.0
    """
    #angle = np.arctan2(norm(np.cross(u,v)), np.dot(u,v))
    res = np.sum(u*v) / (norm(u) * norm(v))
    t = np.clip(res,-1.0,1.0)
    angle = np.arccos(t)
    return angle
# like this:  w,x,y,z
# TODO: optimize with lru_cache etc...

class quaternion(np.ndarray):
    def __new__(cls, input_array=[1, 0, 0, 0]):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # Finally, we must return the newly created object:
        return obj

    @classmethod
    def fromaxang(cls, axis, angle):
        """return quaternion from axis-angle"""
        u = normalized(axis)
        phi = angle / 2.0
        a = cos(phi)
        b = sin(phi)
        q = cls([a, b * u[0], b * u[1], b * u[2]])
        return q.normalized()

    def ang(self):
        Theta = 2 * atan2(np.sqrt(self[1] * self[1] +
                                  self[2] * self[2] +
                                  self[3] * self[3]),
                          self[0])
        return Theta

    def inv(self):
        return quaternion([self[0],
                           -self[1],
                           -self[2],
                           -self[3]])

    @classmethod
    def fromeu(cls, angs):
        """function that generates quaternions from euler angles

        order of rot matrices: RotZ * RotY' * RotX'
        """
        sp = sin(angs[0] / 2.0)
        sth = sin(angs[1] / 2.0)
        sps = sin(angs[2] / 2.0)
        cp = cos(angs[0] / 2.0)
        cth = cos(angs[1] / 2.0)
        cps = cos(angs[2] / 2.0)
        qs = [cp * cps * cth + sp * sps * sth,
              -cp * sps * sth + cps * cth * sp,
              cp * cps * sth + cth * sp * sps,
              cp * cth * sps - cps * sp * sth]
        return np.asarray(qs).view(cls)

    def remove_ambiguity(self):
        return (1 - 2 * (self[0] < 0)) * self

    @classmethod
    def fromvec(cls, vec):
        """create a pseudo-quaternion (with scalar part w = 0)"""
        return cls([0, *vec])

    @classmethod
    def identity(cls):
        # could also be cls.fromeu([0,0,0])
        return cls([1, 0, 0, 0])

    def conjugate(self):
        return quaternion([self[0], -self[1], -self[2], -self[3]])

    def toeu(self):
        """convert into an euler cosine rotation matrix"""
        eus = [atan2(2 *
                     self[0] *
                     self[1] +
                     2 *
                     self[2] *
                     self[3], -
                     2 *
                     self[1]**2 -
                     2 *
                     self[2]**2 +
                     1), asin(2 *
                              self[0] *
                              self[2] -
                              2 *
                              self[1] *
                              self[3]), atan2(2 *
                                              self[0] *
                                              self[3] +
                                              2 *
                                              self[1] *
                                              self[2], -
                                              2 *
                                              self[2]**2 -
                                              2 *
                                              self[3]**2 +
                                              1)]
        return np.asarray(eus)

    def __matmul__(self, quat):
        """define @-operator as a vector product"""
        #return quaternion(self.m4() @ quat)
        q = quat
        sq = self
        return quaternion(((sq[0]*q[0] - sq[1]*q[1] - sq[2]*q[2] - sq[3]*q[3]),
                           (sq[0]*q[1] + sq[1]*q[0] + sq[2]*q[3] - sq[3]*q[2]),
                           (sq[0]*q[2] - sq[1]*q[3] + sq[2]*q[0] + sq[3]*q[1]),
                           (sq[0]*q[3] + sq[1]*q[2] - sq[2]*q[1] + sq[3]*q[0])))

    def m4(self):
        """return matrix representation of quaternion"""
        q0, q1, q2, q3 = self[0], self[1], self[2], self[3]
        m4 = [[q0, q1, q2, q3],
              [-q1, q0, -q3, q2],
              [-q2, q3, q0, -q1],
              [-q3, -q2, q1, q0]]
        return np.array(m4)

    def norm(self):
        return np.linalg.norm(self)

    def normalized(self):
        return self / self.norm()

    def normalized2(self):
        newq = quaternion(np.copy(self))
        newq[0] = np.sqrt(1 - np.dot(newq[1:], newq[1:]))
        return newq

    def qvprod(self, vec):
        """
        something is wrong here...
        
        [1,2,3,4]       [a]
        [1,2,3,4]       [b]
        [1,2,3,4]  x    [c]
        [0,0,0,1]       [0]
        """
        raise NotImplementedError
        
        return self.m4()[1:, 1:] @ vec

    def qprod(self, quat):
        """multiply with another quaternion"""
        return quaternion(self.m4() @ quat)

    def dcm(self):
        return self.rmat()

    def rmat(self):
        """return rotation matrix"""
        a, b, c, d = self[0], self[1], self[2], self[3]  # w,x,y,z
        a2, b2, c2, d2 = a * a, b * b, c * c, d * d
        e = 2 * a * d
        f = 2 * b * c
        R = [[a2 + b2 - c2 - d2, -e + f, 2 * a * c + 2 * b * d],
             [e + f, a2 - b2 + c2 - d2, -2 * a * b + 2 * c * d],
             [-2 * a * c + 2 * b * d, 2 * a * b + 2 * c * d, a2 - b2 - c2 + d2]]
        return np.array(R)

    def __repr__(self):
        return 'quaternion' + str(self)


def normalize_quaternion_series(series):
    return (1 - 2 * (series[:, 0] < 0).reshape(-1, 1)) * series


#TODO: put this into the quaternion class
def getquatrot(u, v):
    """get quaternion rotation (shortest path) from one vector into another (from u into v)

    returns w,x,y,z
    """
    u, v = np.array(u[:]), np.array(v[:])
    k_cos_theta = np.dot(u, v)
    u2 = np.sum(u * u)
    v2 = np.sum(v * v)
    k = np.sqrt(u2 * v2)

    if (k_cos_theta / k == -1):
        # print("180!!!!")
        # 180 degree rotation around any orthogonal vector
        vx = ortho(u)
        # print(vx)
        x, y, z = vx / np.sqrt(np.sum(vx * vx))
        return quaternion([0, x, y, z])

    w = k_cos_theta + k
    x, y, z = np.cross(u, v)
    return quaternion(normalized(np.array([w, x, y, z])))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
