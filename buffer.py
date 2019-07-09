import torch
import numpy as np
import scipy.signal
from utils.mpi_tools import mpi_statistics_scalar

class Buffer(object):
    def __init__(self, obs_dim, act_dim, batch_size, ep_len, dc_interv, gamma=0.99, lam=0.95, N=11):
        self.max_batch = batch_size
        self.dc_interv = dc_interv
        self.max_s = batch_size * ep_len
        self.obs_dim = obs_dim

        self.obs = np.zeros((self.max_s, obs_dim))
        self.act = np.zeros((self.max_s, act_dim))
        self.rew = np.zeros(self.max_s)
        self.pos = np.zeros(self.max_s) # The possibity that dc make it wrong
        self.end = np.zeros(batch_size + 1) # The first will always be 0

        self.ptr = 0
        self.eps = 0
        self.dc_eps = 0

        self.N = N
        self.dc_buff = np.zeros((self.max_batch * self.dc_interv, self.N - 1, obs_dim))

        self.gamma = gamma
        self.lam = lam

    def calc_diff(self):
        # Store the differences of states into memory when epsiode ends
        start = int(self.end[self.eps])
        ep_l = self.ptr - start - 1
        for i in range(self.N-1):
            prev = int(i*ep_l/(self.N-1))
            succ = int((i+1)*ep_l/(self.N-1))
            self.dc_buff[self.dc_eps, i] = self.obs[start + succ][:self.obs_dim] - self.obs[start + prev][:self.obs_dim]

        return self.dc_buff[self.dc_eps]

    def retrieve_dc_buff(self):
        assert self.dc_eps == self.max_batch * self.dc_interv
        self.dc_eps = 0
        return self.dc_buff

class BufferS(Buffer):
    def __init__(self, obs_dim, act_dim, batch_size, ep_len, dc_interv, gamma=0.99, lam=0.95, N=11):
        super(BufferS, self).__init__(obs_dim, act_dim, batch_size, ep_len, dc_interv, gamma, lam, N)
        self.ret = np.zeros(self.max_s)
        self.adv = np.zeros(self.max_s)
        self.lgp = np.zeros(self.max_s)
        self.val = np.zeros(self.max_s)
        self.ent = np.zeros(self.max_s) # Entropy

    def store(self, obs, act, rew, val, lgp):
        assert self.ptr < self.max_s
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.val[self.ptr] = val
        self.lgp[self.ptr] = lgp
        self.ptr += 1

    def end_episode(self, pret_pos, last_val=0): # pret_pos gives the log possibility of cheating the discriminator
        ep_slice = slice(int(self.end[self.eps]), self.ptr)
        rewards = np.append(self.rew[ep_slice], last_val)
        values = np.append(self.val[ep_slice], last_val)
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        returns = scipy.signal.lfilter([1], [1, float(-self.gamma)], rewards[::-1], axis=0)[::-1]
        self.ret[ep_slice] = returns[:-1]
        self.adv[ep_slice] = scipy.signal.lfilter([1], [1, float(-self.gamma * self.lam)], deltas[::-1], axis=0)[::-1]
        self.pos[ep_slice] = pret_pos

        self.eps += 1
        self.dc_eps += 1
        self.end[self.eps] = self.ptr

    def retrieve_all(self):
        assert self.eps == self.max_batch
        occup_slice = slice(0, self.ptr)
        self.ptr = 0
        self.eps = 0
        adv_mean, adv_std = mpi_statistics_scalar(self.adv[occup_slice])
        pos_mean, pos_std = mpi_statistics_scalar(self.pos[occup_slice])
        self.adv[occup_slice] = (self.adv[occup_slice] - adv_mean) / adv_std
        self.pos[occup_slice] = (self.pos[occup_slice] - pos_mean) / pos_std
        return [self.obs[occup_slice], self.act[occup_slice], self.adv[occup_slice], self.pos[occup_slice],
            self.ret[occup_slice], self.lgp[occup_slice]]

class BufferT(Buffer):
    def __init__(self, obs_dim, act_dim, batch_size, ep_len, dc_interv, gamma=0.99, lam=0.95, N=11):
        super(BufferT, self).__init__(obs_dim, act_dim, batch_size, ep_len, dc_interv, gamma, lam, N)

    def store(self, obs, act, rew):
        assert self.ptr < self.max_s
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.ptr += 1

    def end_episode(self, pret_pos):
        ep_slice = slice(int(self.end[self.eps]), self.ptr)
        self.pos[ep_slice] = pret_pos

        self.eps += 1
        self.dc_eps += 1
        self.end[self.eps] = self.ptr

    def retrieve_all(self):
        self.ptr = 0
        self.eps = 0