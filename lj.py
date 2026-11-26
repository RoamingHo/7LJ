"""LJ in minimum-distance units: v(r) = r**(-12) - 2*r**(-6)."""
import numpy as np


def energy_gradient(xyz):
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    i, j = np.triu_indices(len(xyz), 1)
    delta = xyz[i] - xyz[j]
    q = np.einsum('ij,ij->i', delta, delta)
    if np.any(q <= 0):
        raise ValueError('Coincident atoms have infinite LJ energy')
    t = q**-3
    energy = np.sum(t*t - 2*t)
    pair_gradient = (12*(t-t*t)/q)[:, None] * delta
    gradient = np.zeros_like(xyz)
    np.add.at(gradient, i, pair_gradient)
    np.add.at(gradient, j, -pair_gradient)
    return float(energy), gradient


def hessian(xyz):
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    matrix = np.zeros((xyz.size, xyz.size))
    for i in range(len(xyz)):
        for j in range(i):
            delta = xyz[i] - xyz[j]
            q = delta @ delta
            if q <= 0:
                raise ValueError('Coincident atoms have infinite LJ energy')
            block = 12*(q**-4-q**-7)*np.eye(3) + (168*q**-8-96*q**-5)*np.outer(delta, delta)
            si, sj = slice(3*i, 3*i+3), slice(3*j, 3*j+3)
            matrix[si, si] += block
            matrix[sj, sj] += block
            matrix[si, sj] -= block
            matrix[sj, si] -= block
    return matrix


def seed_geometry(n):
    if n == 4:
        return np.array([[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]]) / np.sqrt(8.)
    if n == 6:
        return np.vstack([np.eye(3), -np.eye(3)]) / np.sqrt(2.)
    if n in (5, 7):
        k = n-2
        radius = 1/(2*np.sin(np.pi/k))
        height = np.sqrt(1-radius*radius)
        theta = 2*np.pi*np.arange(k)/k
        ring = np.column_stack([radius*np.cos(theta), radius*np.sin(theta), np.zeros(k)])
        return np.vstack([ring, [0., 0., height], [0., 0., -height]])
    raise ValueError('Seed geometry is defined for N=4,5,6,7')


def canonicalize(xyz):
    """Diameter endpoints first, then x-sorted atoms; quotient rigid motions."""
    xyz = np.asarray(xyz, dtype=float).copy()
    distances = np.linalg.norm(xyz[:, None]-xyz[None, :], axis=-1)
    a, b = np.unravel_index(np.argmax(distances), distances.shape)
    if distances[a, b] == 0:
        raise ValueError('Need distinct atoms')
    xyz -= xyz[a].copy()
    ex = xyz[b]/distances[a, b]
    rest = [i for i in range(len(xyz)) if i not in (a, b)]
    rest.sort(key=lambda i: xyz[i] @ ex)
    ey = xyz[rest[0]] - (xyz[rest[0]] @ ex)*ex
    if np.linalg.norm(ey) < 1e-12:
        trial = np.eye(3)[np.argmin(abs(ex))]
        ey = trial - (trial @ ex)*ex
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    out = xyz[[a, b]+rest] @ np.column_stack([ex, ey, ez])
    if len(out) > 3 and out[3, 2] < 0:
        out[:, 2] *= -1
    return out
