import numpy as np
import matplotlib.pyplot as plt

class Cube:
    def __init__(self, d, start=None, end=None):
        if start is None:
            self.start = -np.ones(d)
        else:
            self.start = start
        if end  is None:
            self.end = np.ones(d)
        else:
            self.end = end

        self.d = d

    def sample_point(self):
        return np.random.uniform(self.start, self.end)

    def transform(self, T):
        # TODO: we assume not rotations....
        center = (self.end+self.start)/2.
        scale = (self.end-self.start)/2.
        return Cube(self.d, T((self.start-center)/scale)*scale + center, T((self.end-center)/scale)*scale + center)

    def is_inside(self, pt):
        return np.all(self.start <= pt) and np.all(pt <= self.end)

class Triangle:
    def __init__(self, d, pts=None):
        assert d == 2
        if pts is None:
            self.pts = np.array([[-1.,-1.], [0.,1.], [1.,-1.]])
        else:
            self.pts = pts

        self.d = d

    def sample_point(self):
        while True:
            x = np.random.uniform(np.min(self.pts, axis=0), np.max(self.pts, axis=0))
            if self.is_inside(x):
                return x

    def transform(self, T):
        # TODO: we assume not rotations....
        center = (np.min(self.pts, axis=0)+np.max(self.pts, axis=0))/2. 
        scale = (self.pts[2,0]-self.pts[0,0])/2.
        return Triangle(self.d, T((self.pts-center)/scale)*scale + center)

    def is_inside(self, pt):
        if pt[0] < self.pts[0,0] or pt[0] > self.pts[2,0] or pt[1] < self.pts[0,1]:
            return False

        if pt[0] <= self.pts[1,0]:
            a = (self.pts[0,1]-self.pts[1,1])/(self.pts[0,0]-self.pts[1,0])
            b = self.pts[0,1]-a*self.pts[0,0]
        else:
            a = (self.pts[1,1]-self.pts[2,1])/(self.pts[1,0]-self.pts[2,0])
            b = self.pts[1,1]-a*self.pts[1,0]

        return a*pt[0] + b >= pt[1]



def rand_ifs_labeled(cur_shape, max_depth, cur_depth, prob_vec, ifs):
    prob = prob_vec[cur_depth-1]
    if cur_depth == 0:
        x = cur_shape.sample_point()
        return x, True
    else:
        next_shapes = [cur_shape.transform(T) for T in ifs]
        if np.random.binomial(1, prob) == 1:
            found_point = False
            while True:
                x = cur_shape.sample_point()
                if not np.any([shape.is_inside(x) for shape in next_shapes]):
                    found_point = True
                    break
            return x, False
        else:
            next_shape = np.random.choice(next_shapes)
        return rand_ifs_labeled(next_shape, max_depth, cur_depth-1, prob_vec, ifs)


def get_ifs_set(m,d,depth,prob_vec,ifs,shape):
    X = []
    Y = []
    print 'generating %d examples...' % m
    for i in range(m):
        x,y = rand_ifs_labeled(shape, depth, depth, prob_vec,ifs)
        X.append(x)
        Y.append(y)

    X = np.stack(X)
    Y = np.stack(Y)
    Y = Y*2-1 # to +/- 1
    Y = np.expand_dims(Y, axis=1)

    return X,Y

def get_sierpinsky_set(m,d,depth,prob_vec):
    shape = Cube(d)
    b1 = np.array([0., 0.5])
    b2 = np.array([-0.5, -0.5])
    b3 = np.array([0.5, -0.5])
    sierpinsky = [lambda x: x/2.0+b1, lambda x: x/2.0+b2, lambda x: x/2.0+b3]
    return get_ifs_set(m,d,depth,prob_vec,sierpinsky,shape)

def get_pentaflake_set(m,d,depth,prob_vec):
    shape = Cube(d)
    b1 = np.array([-2./3., 0.])
    b2 = np.array([2./3., 0.])
    b3 = np.array([0., 2./3.])
    b4 = np.array([-1./3., -2./3.])
    b5 = np.array([1./3., -2./3.])
    pentaflake = [lambda x: x/3.0+b1, lambda x: x/3.0+b2, lambda x: x/3.0+b3, lambda x: x/3.0+b4, lambda x: x/3.0+b5]
    return get_ifs_set(m,d,depth,prob_vec,pentaflake,shape)

def get_vicsek_set(m,d,depth,prob_vec):
    shape = Cube(d)
    b1 = np.array([-2./3., -2./3.])
    b2 = np.array([-2./3., 2./3.])
    b3 = np.array([2./3., -2./3.])
    b4 = np.array([2./3., 2./3.])
    b5 = np.array([0., 0.])
    vicsek = [lambda x: x/3.0+b1, lambda x: x/3.0+b2, lambda x: x/3.0+b3, lambda x: x/3.0+b4, lambda x: x/3.0+b5]
    return get_ifs_set(m,d,depth,prob_vec,vicsek,shape)


if __name__ == "__main__":
    d = 2
    shape = Cube(d)
    depth = 5
    pdeg = 0
    prob_vec = np.linspace(depth,1,depth)**pdeg
    prob_vec /= np.sum(prob_vec)
    prob_vec = 1-0.5**prob_vec

    b1 = np.array([-2./3., -2./3.])
    b2 = np.array([2./3., -2./3.])
    b3 = np.array([-2./3., 2./3.])
    b4 = np.array([2./3., 2./3.])
    cantor = [lambda x: x/3.0+b1, lambda x: x/3.0+b2, lambda x: x/3.0+b3, lambda x: x/3.0+b4]

    b1 = np.array([0., 0.5])
    b2 = np.array([-0.5, -0.5])
    b3 = np.array([0.5, -0.5])
    sierpinsky = [lambda x: x/2.0+b1, lambda x: x/2.0+b2, lambda x: x/2.0+b3]

    b1 = np.array([-2./3., 0.])
    b2 = np.array([2./3., 0.])
    b3 = np.array([0., 2./3.])
    b4 = np.array([-1./3., -2./3.])
    b5 = np.array([1./3., -2./3.])
    pentaflake = [lambda x: x/3.0+b1, lambda x: x/3.0+b2, lambda x: x/3.0+b3, lambda x: x/3.0+b4, lambda x: x/3.0+b5]

    b1 = np.array([-2./3., -2./3.])
    b2 = np.array([-2./3., 2./3.])
    b3 = np.array([2./3., -2./3.])
    b4 = np.array([2./3., 2./3.])
    b5 = np.array([0., 0.])
    vicsek = [lambda x: x/3.0+b1, lambda x: x/3.0+b2, lambda x: x/3.0+b3, lambda x: x/3.0+b4, lambda x: x/3.0+b5]

    X,Y = get_ifs_set(10000, d, depth, prob_vec, pentaflake, shape)
    print np.sum(Y == 1)/10000.0
    plt.scatter(X[Y[:,0]==1,0], X[Y[:,0]==1,1], c='r', s=5.)
    plt.scatter(X[Y[:,0]==-1,0], X[Y[:,0]==-1,1], c='b', s=5.)
    plt.show()
