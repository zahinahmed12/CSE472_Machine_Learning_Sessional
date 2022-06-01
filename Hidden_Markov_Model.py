# import matplotlib.pyplot as plt
# import pandas as pd
from hmmlearn import hmm
import numpy as np
import math
import random

EPS = 1e-20

with open('./Sample input and output for HMM/Input/data.txt') as f:
    lines = f.readlines()

y = [float(x) for x in [line.strip() for line in lines]]
T = len(y)
f.close()

with open('./Sample input and output for HMM/Input/parameters.txt.txt') as f:
    lines2 = f.readlines()
f.close()

lines2 = [line.strip() for line in lines2]

no_of_states = int(lines2[0])
trans_mat = []
for i in range(no_of_states):
    r = [float(x) for x in lines2[i+1].split('\t')]
    trans_mat.append(r)

# print(trans_mat)
means = [float(x) for x in lines2[no_of_states+1].split('\t')]
variances = [float(x) for x in lines2[no_of_states+2].split('\t')]
# print(variances)


def new_log(x):
    if x == 0.0:
        x = 1.0
    return math.log(x)


def get_emission_prob(ly, u, s):
    # print(s)
    d = 1/math.sqrt(2*math.pi*s)
    e = -0.5/s
    return [d*math.exp(e*(x-u)**2) for x in ly]


def get_initial_prob(ns, t):
    b = np.zeros(ns-1)
    b = np.append(b, [1], axis=0)
    # print(b)
    a = []
    for itr in range(ns-1):
        a.append([row[itr] for row in t])
    a = np.array(a)
    np.fill_diagonal(a, a.diagonal()-1)
    c = np.ones(ns)
    a = np.append(a, [c], axis=0)
    # print(a)

    return np.linalg.solve(a, b)


def viterbi(ns, t, m, u, ly, ts):
    mat = np.full((ns, ts), -999999999.0)
    init_prob = get_initial_prob(ns, t)

    emission_mat = []
    for itr in range(ns):
        em = get_emission_prob(ly, m[itr], u[itr])
        emission_mat.append(em)
    emission_mat = np.array(emission_mat)

    res = np.full(ts, -1)
    prev_state = np.full((ns, ts), -1)
    for step in range(ts):
        for row in range(ns):
            for st in range(ns):
                if step == 0:
                    v = new_log(init_prob[st]) + new_log(t[st][row]) + new_log(emission_mat[row, step])
                else:
                    v = mat[st, step-1] + new_log(t[st][row]) + new_log(emission_mat[row, step])
                if v > mat[row, step]:
                    mat[row, step] = v
                    prev_state[row, step] = st

    ind = np.argmax(mat, axis=0)
    last_state = ind[ts-1]
    res[ts-1] = last_state

    for col in range(ts, 1, -1):
        res[col-2] = prev_state[last_state, col-1]
        last_state = prev_state[last_state, col-1]

    return res


def climate_pattern(es, filename):
    out = open(filename, "w")

    for itr in range(T):
        if es[itr] == 0:
            out.write('"El Nino"\n')
        elif es[itr] == 1:
            out.write('"La Nina"\n')
        else:
            out.write("?\n")
    out.close()


def random_trans_mat(ns):
    trans = np.full((ns, ns), 0.0)
    for row in range(ns):
        for col in range(ns):
            random.seed(row+col+3)
            trans[row, col] = random.random()
            if trans[row, col] < 0.01:
                trans[row, col] = 0.2
        sum_row = trans[row].sum()
        trans[row] = np.true_divide(trans[row], sum_row)
    return trans


# print(random_trans_mat(no_of_states))


def get_mean_from_data(dl):
    dl = np.array(dl)
    return np.mean(dl)


# print(get_mean_from_data(y))


def get_variance_from_data(dl):
    dl = np.array(dl)
    N = len(dl)
    mean = get_mean_from_data(dl)
    dl = dl - mean
    dl = np.multiply(dl, dl)
    return np.sum(dl)/N


# print(get_variance_from_data(y))


def get_random_emission(ns, ly):
    mean = get_mean_from_data(ly)
    variance = get_variance_from_data(ly)

    m = []
    u = []

    for itr in range(1, ns+1):
        u.append(variance+itr)
        if itr % 2 == 1:
            m.append(mean + itr*15)
        else:
            m.append(mean - itr*10)

    return m, u


def forward_mat(ns, ly, trans, init_prob, m, u):
    ts = len(ly)
    mat = np.full((ns, ts), 0.0)
    emission = []
    for itr in range(ns):
        em = get_emission_prob(ly, m[itr], u[itr])
        emission.append(em)
    emission = np.array(emission)

    for step in range(ts):
        for row in range(ns):
            sm = 0
            for st in range(ns):
                if step == 0:
                    v = init_prob[st] * trans[st][row] * emission[row, step]
                else:
                    v = mat[st, step-1] * trans[st][row] * emission[row, step]
                sm += v
            mat[row, step] = sm
        col_total = 0.0
        for row in range(ns):
            mat[row, step] += EPS
            col_total += mat[row, step]
        for row in range(ns):
            mat[row, step] = mat[row, step]/col_total

    # with open('debug.txt', 'w') as out:
    #     for step in range(ts):
    #         for row in range(ns):
    #             out.write('{:.6f} '.format(mat[row, step]))
    #         out.write('\n')
    return mat


# frd = forward_mat(no_of_states, y, trans_mat, st_prob, means, variances)
# forward_mat(no_of_states, y, trans_est, st_prob, rand_mean, rand_var)


def backward_mat(ns, ly, trans, m, u):
    ts = len(ly)
    mat = np.full((ns, ts), 0.0)
    emission = []
    for itr in range(ns):
        em = get_emission_prob(ly, m[itr], u[itr])
        emission.append(em)
    emission = np.array(emission)

    for row in range(ns):
        mat[row, ts-1] = 1.0

    for step in range(ts-2, -1, -1):
        for row in range(ns):
            sm = 0
            for st in range(ns):
                v = mat[st, step+1] * trans[row][st] * emission[st, step+1]
                sm += v
            mat[row, step] = sm
        col_total = 0.0
        for row in range(ns):
            mat[row, step] += EPS
            col_total += mat[row, step]
        for row in range(ns):
            mat[row, step] = mat[row, step]/col_total

    # with open('debug.txt', 'w') as out:
    #     for step in range(ts):
    #         for row in range(ns):
    #             out.write('{:.6f} '.format(mat[row, step]))
    #         out.write('\n')
    return mat


# bck = backward_mat(no_of_states, y, trans_mat, means, variances)
# backward_mat(no_of_states, y, trans_est, rand_mean, rand_var)


def get_forward_sink(forward):
    ns = forward.shape[0]
    ts = forward.shape[1]
    s = 0.0
    for row in range(ns):
        s += forward[row, ts-1]
    return s


def get_matrix_a(forward, backward):
    ns = forward.shape[0]
    ts = forward.shape[1]
    fs = get_forward_sink(forward)

    a = np.full((ns, ts), 0.0)
    for step in range(ts):
        for row in range(ns):
            a[row, step] = forward[row, step] * backward[row, step] / fs
        col_total = 0.0
        for row in range(ns):
            a[row, step] += EPS
            col_total += a[row, step]
        for row in range(ns):
            a[row, step] /= col_total

    # with open('debug.txt', 'w') as out:
    #     for step in range(ts):
    #         for row in range(ns):
    #             out.write('{:.6f} '.format(a[row, step]))
    #         out.write('\n')

    return a


def get_mean_nd_variance(ly, a):
    ns = a.shape[0]
    # ts = a.shape[1]
    sm = a.sum(axis=1)
    ly = np.array(ly)
    ms = []
    vs = []
    for row in range(ns):
        ms.append(np.sum(np.multiply(a[row], ly))/sm[row])

    for row in range(ns):
        data = ly - ms[row]
        data = np.multiply(data, data)
        vs.append(np.sum(np.multiply(a[row], data))/sm[row])

    # print(ms)
    # print(vs)
    return ms, vs


# get_mean_nd_variance(y, get_matrix_a(frd, bck))


def get_matrix_b(trans, ly, m, u, forward, backward):
    ns = forward.shape[0]
    ts = forward.shape[1]
    fs = get_forward_sink(forward)
    # print(fs)
    b = np.full((ns, ns, ts-1), 0.0)
    # print(b)
    emission = []
    for itr in range(ns):
        em = get_emission_prob(ly, m[itr], u[itr])
        emission.append(em)
    emission = np.array(emission)

    for step in range(ts-1):
        for row in range(ns):
            for col in range(ns):
                b[row, col, step] = forward[row, step] * trans[row][col] * backward[col, step+1] * emission[col, step+1]
                b[row, col, step] /= fs
        sm = 0.0
        for row in range(ns):
            for col in range(ns):
                b[row, col, step] += EPS
                sm += b[row, col, step]
        for row in range(ns):
            for col in range(ns):
                b[row, col, step] /= sm

    # with open('debug.txt', 'w') as out:
    #     for step in range(ts-1):
    #         for row in range(ns):
    #             for col in range(ns):
    #                 out.write('{:.6f} '.format(b[row, col, step]))
    #             out.write('\n')
    #         out.write('\n')
    # with open('debug.txt', 'w') as out:
    #     for step in range(ts):
    #         for row in range(ns):
    #             out.write('{:.6f} '.format(emission[row, step]))
    #         out.write('\n')
    # print(trans)
    # print(m)
    # print(u)

    return b


def get_transition(b):
    ts = b.shape[2]
    ns = b.shape[0]
    new_trans = np.full((ns, ns), 0.0)
    for step in range(ts):
        for row in range(ns):
            for col in range(ns):
                new_trans[row, col] += b[row, col, step]
    for row in range(ns):
        new_trans[row] += EPS
        sum_row = new_trans[row].sum()
        new_trans[row] = np.true_divide(new_trans[row], sum_row)

    # print(new_trans)
    return new_trans


# get_transition(get_matrix_b(trans_mat, y, means, variances, frd, bck))


def e_step(ns, ly, trans, init_prob, m, u):
    forward = forward_mat(ns, ly, trans, init_prob, m, u)
    backward = backward_mat(ns, ly, trans, m, u)

    return forward, backward


def m_step(ly, trans, m, u, forward, backward):
    a = get_matrix_a(forward, backward)
    b = get_matrix_b(trans, ly, m, u, forward, backward)
    m, u = get_mean_nd_variance(ly, a)
    trans = get_transition(b)
    return trans, m, u


def baum_welch(ns, ly, trans, m, u, max_itr=10):
    for itr in range(max_itr):
        init_prob = get_initial_prob(ns, trans)
        forward, backward = e_step(ns, ly, trans, init_prob, m, u)
        trans, m, u = m_step(ly, trans, m, u, forward, backward)

    return trans, m, u


def print_param(ns, trans, m, u, init_prob):
    with open('param_learned.txt', 'w') as out:
        out.write(f'{ns}')
        out.write('\n')
        for row in range(ns):
            for col in range(ns):
                out.write('{:.7f} '.format(trans[row, col]))
            out.write('\n')
        for state in range(ns):
            out.write('{:.4f} '.format(m[state]))
        out.write('\n')
        for state in range(ns):
            out.write('{:.6f} '.format(u[state]))
        out.write('\n')
        for state in range(ns):
            out.write('{:.2f} '.format(init_prob[state]))
        out.write('\n')
        out.close()


def compare_estimation(ns, ly, trans, m, u, es):

    ly = np.array(ly).reshape((-1, 1))
    init_prob = np.array(get_initial_prob(ns, trans))
    trans = np.array(trans)
    m = np.array(m).reshape((ns, 1))
    u = np.array(u).reshape((ns, 1, 1))
    # print(ly.shape)

    model = hmm.GaussianHMM(n_components=ns, covariance_type='full', init_params='')

    model.startprob_ = init_prob  # (2,)
    model.transmat_ = trans  # (n, n)
    model.means_ = m  # (n, 1)
    model.covars_ = u  # (n, 1, 1)

    seq = model.predict(ly)  # (-1, 1)
    print(f'No. of differences with "sci-kit hmmlearn" estimation of states before learning: {np.sum(seq != es)}')


def compare_learned_estimation(ns, ly, trans, m, u, es):
    ly = np.array(ly).reshape(-1, 1)
    init_prob = np.array(get_initial_prob(ns, trans))
    trans = np.array(trans)
    m = np.array(m).reshape(ns, 1)
    u = np.array(u).reshape((ns, 1, 1))

    model = hmm.GaussianHMM(n_components=ns, covariance_type='full', init_params='')

    model.startprob_ = init_prob  # (2,)
    model.transmat_ = trans  # (n, n)
    model.means_ = m  # (n, 1)
    model.covars_ = u  # (n, 1, 1)

    model.fit(ly)  # (-1, 1)

    print(f'\nParameters learned by hmmlearn:')
    print(f'Transition matrix:')
    print(model.transmat_)
    print(f'\nMeans:')
    print(model.means_)
    print(f'\nVariances:')
    print(model.covars_)

    seq = model.predict(ly)  # (-1, 1)
    print(f'\nNo. of differences with "sci-kit hmmlearn" estimation of states after learning: {np.sum(seq != es)}')


trans_est = random_trans_mat(no_of_states)
st_prob = get_initial_prob(no_of_states, trans_mat)
rand_mean, rand_var = get_random_emission(no_of_states, y)
learned_trans, learned_m, learned_u = baum_welch(no_of_states, y, trans_mat, means, variances)
# learned_trans, learned_m, learned_u = baum_welch(no_of_states, y, trans_est, rand_mean, rand_var)


estimated_states = viterbi(no_of_states, trans_mat, means, variances, y, T)
climate_pattern(estimated_states, "viterbi_before_learning.txt")
learned_states = viterbi(no_of_states, learned_trans, learned_m, learned_u, y, T)
climate_pattern(learned_states, "viterbi_after_learning.txt")
print_param(no_of_states, learned_trans, learned_m, learned_u, st_prob)
# compare_estimation(no_of_states, y, trans_mat, means, variances, estimated_states)
# compare_learned_estimation(no_of_states, y, learned_trans, learned_m, learned_u, learned_states)
