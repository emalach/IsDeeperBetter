import numpy as np
import matplotlib.pyplot as plt

def in_cantor_d(x, cur_cube, cur_depth, factor=0.3333):
    if cur_depth == 0:
        return True
    else:
        next_cube = []
        for dim,rng in enumerate(cur_cube):
            rng_len = rng[1]-rng[0]
            assert (x[dim] >= rng[0]) and (x[dim] <= rng[1])
            if x[dim] <= rng[0] + rng_len*factor:
                next_cube.append((rng[0], rng[0] + rng_len*factor))
            elif x[dim] >= rng[1]-rng_len*factor:
                next_cube.append((rng[1]-rng_len*factor, rng[1]))
            else:
                return False
        return in_cantor_d(x, next_cube, cur_depth-1, factor)

def rand_cantor_d(cur_cube, cur_depth, factor=0.3333, last_choice=None):
    if cur_depth == 0:
        cur_cube = np.array(cur_cube)
        x = np.random.uniform(cur_cube[:,0], cur_cube[:,1])
        return x, last_choice
    else:
        next_cube = []
        choices = []
        for dim,rng in enumerate(cur_cube):
            rng_len = rng[1]-rng[0]
            choice = np.random.choice(2)
            if choice == 0:
                next_cube.append((rng[0], rng[0] + rng_len*factor))
            else:
                next_cube.append((rng[1]-rng_len*factor, rng[1]))
            choices.append(choice)
        return rand_cantor_d(next_cube, cur_depth-1,factor,choices)

def rand_cantor_d_labeled(cur_cube, max_depth, cur_depth, prob_vec, factor=0.3333):
    #prob = 1-0.5**float(1.0/max_depth)
    prob = prob_vec[cur_depth-1]
    if cur_depth == 0:
        cur_cube = np.array(cur_cube)
        x = np.random.uniform(cur_cube[:,0], cur_cube[:,1])
        return x, True
    else:
        if np.random.binomial(1, prob) == 1:
            cur_cube = np.array(cur_cube)
            rng_len = cur_cube[:,1]-cur_cube[:,0]
            next_cube = np.stack((cur_cube[:,0]+factor*rng_len, cur_cube[:,1]-factor*rng_len),axis=1)
            x = np.random.uniform(next_cube[:,0], next_cube[:,1])
            return x, False
        else:
            next_cube = []
            for dim,rng in enumerate(cur_cube):
                rng_len = rng[1]-rng[0]
                choice = np.random.choice(2)
                if choice == 0:
                    next_cube.append((rng[0], rng[0] + rng_len*factor))
                else:
                    next_cube.append((rng[1]-rng_len*factor, rng[1]))
        return rand_cantor_d_labeled(next_cube, max_depth, cur_depth-1, prob_vec,factor)


def get_cantor_set(m,d,depth,prob_vec=None,factor=0.3333, mode='fractal_uniform'):
    if prob_vec is None:
        prob_vec = [1-0.5**float(1.0/depth)]*depth
    cube = [(-1,1)]*d
    X = []
    Y = []
    print 'generating %d examples...' % m
    for i in range(m):
        if mode == 'uniform':
            y = True
            while y is True:
                x = np.random.uniform(-1,1,size=d)
                y = in_cantor_d(x, cube, depth, factor)
            X.append(x)
            Y.append(y)

            x, choices = rand_cantor_d(cube, depth, factor)
            y = True
            X.append(x)
            Y.append(y)
        if mode == 'fractal_uniform':
            x,y = rand_cantor_d_labeled(cube, depth, depth, prob_vec, factor)
            X.append(x)
            Y.append(y)
        elif mode == 'binary':
            x, choices = rand_cantor_d(cube, depth, factor)
            y = np.sum(choices) %2 == 0
            X.append(x)
            Y.append(y)

    X = np.stack(X)
    Y = np.stack(Y)
    Y = Y*2-1 # to +/- 1
    Y = np.expand_dims(Y, axis=1)

    return X,Y

if __name__ == "__main__":
    depth = 1
    pdeg = 0
    curve_num=0
    if pdeg < 10: # polynomial with degree pdeg
        prob_vec = np.linspace(depth,1,depth)**pdeg
    else: #uniform on first pdeg-10 entries
        th = pdeg-10
        prob_vec = np.zeros(depth)
        prob_vec[:th+1] = 1.0
    prob_vec /= np.sum(prob_vec)
    prob_vec = 1-0.5**prob_vec

    X,Y = get_cantor_set(10000, 2, depth, prob_vec)
    print np.sum(Y == 1)/10000.0
    plt.scatter(X[Y[:,0]==1,0], X[Y[:,0]==1,1],s=1)
    plt.scatter(X[Y[:,0]==-1,0], X[Y[:,0]==-1,1],s=1)

    plt.figure()
    fig,ax = plt.subplots(1,1)
    rev_prob_vec = prob_vec[::-1]
    profile = [0.5] + list(0.5+np.cumsum([np.prod(1-rev_prob_vec[:i])*rev_prob_vec[i] for i in range(depth)]))
    ax.plot(profile)
    ax.set_title('curve#%d' % (curve_num+1))
    plt.xlabel('j')
    plt.ylabel('P')


    plt.show()
