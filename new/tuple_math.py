def norm(v):
    return sum(x**2 for x in v)**.5

def dot(a, b):
    return sum(x*y for x, y in zip(a, b))

def cross(a, b):
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])