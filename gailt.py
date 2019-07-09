# Main entrance of GAILT
import numpy as np
import torch
import torch.nn.functional as F 
import gym
import time
from network import Discriminator, ActorCritic, count_vars
from buffer import BufferS, BufferT
from utils.mpi_tools import mpi_fork, proc_id, mpi_statistics_scalar, num_procs
from utils.mpi_torch import average_gradients, sync_all_params
from utils.logx import EpochLogger

def gailt(env_fn, actor_critic=ActorCritic, ac_kwargs=dict(), disc=Discriminator, dc_kwargs=dict(), seed=0, episodes_per_epoch=40,
        epochs=500, gamma=0.99, lam=0.97, pi_lr=3e-5, vf_lr=1e-3, dc_lr=5e-4, train_v_iters=80, train_dc_iters=20, 
        train_dc_interv=5, max_ep_len=1000, logger_kwargs=dict(), save_freq=10):

    l_lam = 0 # balance two loss term

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    ac_kwargs['action_space'] = env.action_space

    # Models
    ac = actor_critic(input_dim=obs_dim[0], **ac_kwargs)
    disc = disc(input_dim=obs_dim[0], **dc_kwargs)

    # TODO: Load expert policy here
    expert = actor_critic(input_dim=obs_dim[0], **ac_kwargs)

    # Buffers
    local_episodes_per_epoch = int(episodes_per_epoch / num_procs())
    buff_s = BufferS(obs_dim[0], act_dim[0], local_episodes_per_epoch, max_ep_len, train_dc_interv)
    buff_t = BufferT(obs_dim[0], act_dim[0], local_episodes_per_epoch, max_ep_len, train_dc_interv)

    # Count variables
    var_counts = tuple(count_vars(module) for module in [ac.policy, ac.value_f, disc.policy])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d, \t d: %d\n'%var_counts)
    
    # Optimizers
    train_pi = torch.optim.Adam(ac.policy.parameters(), lr=pi_lr)
    train_v = torch.optim.Adam(ac.value_f.parameters(), lr=vf_lr)
    train_dc = torch.optim.Adam(disc.policy.parameters(), lr=dc_lr)

    # Parameters Sync
    sync_all_params(ac.parameters())
    sync_all_params(disc.parameters())

    def update(e):
        obs, act, adv, pos, ret, lgp_old = [torch.Tensor(x) for x in buff_s.retrieve_all()]
        buff_t.retrieve_all() # Clear states buffer

        # Policy
        _, lgp, _ = ac.policy(obs, act)
        entropy = (-lgp).mean()

        # Policy loss
        # policy gradient term + entropy term
        print(lgp.mean(), pos.mean())
        pi_loss = -(lgp*pos).mean() - l_lam*entropy
        
        # Train policy
        train_pi.zero_grad()
        pi_loss.backward()
        average_gradients(train_pi.param_groups)
        train_pi.step()

        # Value function
        v = ac.value_f(obs)
        v_l_old = F.mse_loss(v, ret)
        for _ in range(train_v_iters):
            v = ac.value_f(obs)
            v_loss = F.mse_loss(v, ret)

            # Value function train
            train_v.zero_grad()
            v_loss.backward()
            average_gradients(train_v.param_groups)
            train_v.step()

        # Discriminator
        if (e+1) % train_dc_interv == 0:
            s_t = torch.Tensor(buff_s.retrieve_dc_buff()) # trajectories of students
            t_t = torch.Tensor(buff_t.retrieve_dc_buff())
            gt1 = torch.ones(local_episodes_per_epoch * train_dc_interv, dtype=torch.int)
            gt2 = torch.zeros(local_episodes_per_epoch * train_dc_interv, dtype=torch.int)
            _, lgp_s, _ = disc(s_t, gt=gt1)
            _, lgp_t, _ = disc(t_t, gt=gt2)
            dc_loss_old = -lgp_s.mean() - lgp_t.mean()
            for _ in range(train_dc_iters):
                _, lgp_s, _ = disc(s_t, gt=gt1)
                _, lgp_t, _ = disc(t_t, gt=gt2)
                dc_loss = -lgp_s.mean() - lgp_t.mean()

                # Discriminator train
                train_dc.zero_grad()
                dc_loss.backward()
                average_gradients(train_dc.param_groups)
                train_dc.step()

            _, lgp_s, _ = disc(s_t, gt=gt1)
            _, lgp_t, _ = disc(t_t, gt=gt2)
            dc_loss_new = -lgp_s.mean() - lgp_t.mean()
        else:
            dc_loss_old = 0
            dc_loss_new = 0

        # Log the changes
        _, lgp, _, v = ac(obs, act)
        entropy_new = (-lgp).mean()
        pi_loss_new = -(lgp*adv).mean() - l_lam*entropy
        v_loss_new = F.mse_loss(v, ret)
        kl = (lgp_old - lgp).mean()
        intrinsic_ret = pos.mean()
        logger.store(LossPi=pi_loss, LossV=v_l_old, LossDC=dc_loss_old, DeltaLossPi=(pi_loss_new-pi_loss),
            DeltaLossV=(v_loss_new-v_l_old), DeltaLossDC=(dc_loss_new-dc_loss_old), DeltaEnt=(entropy_new-entropy),
            Entropy=entropy, KL=kl, TrueRet=intrinsic_ret)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_t = 0    

    ep_len_t = 0
    for epoch in range(epochs):
        ac.eval()
        disc.eval()
        # We recognize the probability term of index [0] correspond to the teacher's policy
        # Student's policy rollout
        for _ in range(local_episodes_per_epoch):            
            for _ in range(max_ep_len):
                obs = torch.Tensor(o.reshape(1, -1))
                a, _, lopg_t, v_t = ac(obs)

                buff_s.store(o, a.detach().numpy(), r, v_t.item(), lopg_t.detach().numpy())
                logger.store(VVals=v_t)

                o, r, d, _ = env.step(a.detach().numpy()[0])
                ep_ret += r
                ep_len += 1
                total_t += 1

                terminal = d or (ep_len == max_ep_len)
                if terminal:
                    dc_diff = torch.Tensor(buff_s.calc_diff()).unsqueeze(0)
                    _, logp, _ = disc(dc_diff, gt=torch.Tensor([0])) 
                    buff_s.end_episode(logp.detach().numpy())
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # Teacher's policy rollout
        for _ in range(local_episodes_per_epoch):            
            for _ in range(max_ep_len):
                obs = torch.Tensor(o.reshape(1, -1))
                a, _, _, _ = expert(obs)

                buff_t.store(o, a.detach().numpy(), r)

                o, r, d, _ = env.step(a.detach().numpy()[0])
                ep_ret += r
                ep_len += 1
                total_t += 1

                terminal = d or (ep_len == max_ep_len)
                if terminal:
                    dc_diff = torch.Tensor(buff_t.calc_diff()).unsqueeze(0)
                    _, logp, _ = disc(dc_diff, gt=torch.Tensor([1])) 
                    buff_t.end_episode(logp.detach().numpy())
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0


        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, [ac, disc], None)

        # Update
        ac.train()
        disc.train()

        update(epoch)

        # Log
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', total_t)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('LossDC', average_only=True)
        logger.log_tabular('DeltaLossDC', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('DeltaEnt', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.97)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=8)
    parser.add_argument('--episodes-per-epoch', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='gailt')
    args = parser.parse_args()

    mpi_fork(args.cpu)

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    gailt(lambda: gym.make(args.env), actor_critic=ActorCritic, ac_kwargs=dict(hidden_dims=[args.hid]*args.l),
        disc=Discriminator, dc_kwargs=dict(hidden_dims=args.hid), gamma=args.gamma, lam=args.lam, seed=args.seed,
        episodes_per_epoch=args.episodes_per_epoch, epochs=args.epochs, logger_kwargs=logger_kwargs)