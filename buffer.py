import torch
import numpy as np
import scipy.signal
from utils.mpi_tools import mpi_statistics_scalar

class Buffer(object):
    def __init__(self, obs_dim, act_dim, batch_size, ep_len):
        self.max_batch = batch_size
        self.max_volume = batch_size * ep_len
        self.obs_dim = obs_dim

        self.obs = np.zeros((self.max_volume, obs_dim))
        self.act = np.zeros((self.max_volume, act_dim))
        self.rew = np.zeros(self.max_volume)
        self.end = np.zeros(batch_size + 1) # The first term will always be 0 / boundries of trajectories

        self.ptr = 0
        self.eps = 0

    def store(self, obs, act, rew, sdr, val, lgp):
        raise NotImplementedError

    def end_episode(self, last_val=0):
        raise NotImplementedError

    def retrieve_all(self):
        raise NotImplementedError

class BufferS(Buffer):
    def __init__(self, obs_dim, act_dim, batch_size, ep_len, gamma=0.99, lam=0.95):
        super(BufferS, self).__init__(obs_dim, act_dim, batch_size, ep_len)

        self.sdr = np.zeros(self.max_volume) # Pseudo reward, the log prob
        self.ret = np.zeros(self.max_volume) # Discounted return based on self.sdr
        self.val = np.zeros(self.max_volume)
        self.adv = np.zeros(self.max_volume)
        self.lgp = np.zeros(self.max_volume) # Log prob of selected actions, used for entropy estimation

        self.gamma = gamma
        self.lam = lam

    def store(self, obs, act, rew, sdr, val, lgp):
        assert self.ptr < self.max_volume
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.sdr[self.ptr] = sdr
        self.val[self.ptr] = val
        self.lgp[self.ptr] = lgp  
        self.ptr += 1

    def end_episode(self, last_val=0):
        ep_slice = slice(int(self.end[self.eps]), self.ptr)
        rewards = np.append(self.sdr[ep_slice], last_val)
        values = np.append(self.val[ep_slice], last_val)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        returns = scipy.signal.lfilter([1], [1, float(-self.gamma)], rewards[::-1], axis=0)[::-1]
        self.ret[ep_slice] = returns[:-1]
        self.adv[ep_slice] = scipy.signal.lfilter([1], [1, float(-self.gamma * self.lam)], deltas[::-1], axis=0)[::-1]

        self.eps += 1
        self.end[self.eps] = self.ptr

    def retrieve_all(self):
        assert self.eps == self.max_batch
        occup_slice = slice(0, self.ptr)
        self.ptr = 0
        self.eps = 0

        adv_mean, adv_std = mpi_statistics_scalar(self.adv[occup_slice])
        self.adv[occup_slice] = (self.adv[occup_slice] - adv_mean) / adv_std
        return [self.obs[occup_slice], self.act[occup_slice], self.adv[occup_slice], 
            self.ret[occup_slice], self.lgp[occup_slice]]

class BufferT(Buffer):
    def __init__(self, obs_dim, act_dim, batch_size, ep_len, gamma=0.99, lam=0.95):
        super(BufferS, self).__init__(obs_dim, act_dim, batch_size, ep_len, gamma, lam)

    def store(self, obs, act, rew, sdr, val, lgp):
        assert self.ptr < self.max_volume
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew

        self.ptr += 1

    def end_episode(self, last_val=0):
        self.eps += 1
        self.end[self.eps] = self.ptr

    def retrieve_all(self):
        assert self.eps == self.max_batch
        occup_slice = slice(0, self.ptr)
        self.ptr = 0
        self.eps = 0

        return [self.obs[occup_slice], self.act[occup_slice]]