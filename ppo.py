import numpy as np
import torch
from torch.optim import Adam
import time
from dm_control import suite,viewer
import core
from functools import partial
import csv
import tqdm

def write_to_csv(filename, data1, data2):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([data1, data2])

csv_filename = 'data.csv'

with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Reward', 'Timestep'])

class PPOBuffer:

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):

        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):

        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf),np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

def ppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=1000, epochs=5000, gamma=0.99, clip_ratio=0.11, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01):

    seed += 10000
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    U=env.action_spec();act_dim=U.shape[0];
    X=env.observation_spec();obs_dim=14+1+9;

    ac = actor_critic(obs_dim, act_dim, **ac_kwargs)
    #ac.load_state_dict(torch.load('model_params.pth'))

    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    print('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    local_steps_per_epoch = int(steps_per_epoch)
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        approx_kl = (logp_old - logp).mean().item()
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean() * 0.005

        # Useful extra info
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    def update():
        data = buf.get()
        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        ES = False
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info['kl']
            
            if kl > 1.5 * target_kl:
                write_to_csv(csv_filename, data['ret'][-1].item(), i)
                ES = True
                break
            loss_pi.backward()
            pi_optimizer.step()
        
        if ES == False:
            write_to_csv(csv_filename, data['ret'][-1].item(), train_pi_iters)

        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']

    # Main loop: collect experience in env and update/log each epoch
    for epoch in tqdm.tqdm(range(epochs)):
        o, ep_ret, ep_len = env.reset(), 0, 0
        x=o.observation
    
        for t in range(local_steps_per_epoch):
            height_robo = x['height']
            x=np.array(x['orientations'].tolist()+[x['height']]+x['velocity'].tolist())
            a, v, logp = ac.step(torch.from_numpy(x).float().unsqueeze(0))
            o = env.step(a)
            next_x, r, d = o.observation,o.reward,o.last()
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(x, a, r, v, logp)
            
            # Update obs (critical!)
            x = next_x

            timeout = ep_len == max_ep_len
            terminal = d or timeout or height_robo <= 0.5
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:

                if timeout or epoch_ended:
                    x=np.array(x['orientations'].tolist()+[x['height']]+x['velocity'].tolist())
                    _, v, _ = ac.step(torch.from_numpy(x).float().unsqueeze(0))
                else:
                    v = 0
                
                buf.finish_path(v)
                o, ep_ret, ep_len = env.reset(), 0, 0
                x=o.observation
                
        # Perform PPO update!
        update()

    torch.save(ac.state_dict(), 'model_params.pth')
    o=env.reset()
    x=o.observation
    x=np.array(x['orientations'].tolist()+[x['height']]+x['velocity'].tolist())

    def u_policy(dt,x,ac,env):

        with torch.no_grad():
            u=ac.act(torch.from_numpy(x).float().unsqueeze(0))
        o = env.step(u)
        x=o.observation
        x=np.array(x['orientations'].tolist()+[x['height']]+x['velocity'].tolist())

        return u
    
    policy_with_context = partial(u_policy, x=x, ac=ac, env=env)
    viewer.launch(env,policy=policy_with_context)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='walker')
    parser.add_argument('--task', type=str, default='walk')
    parser.add_argument('--hid', type=int, default=128)
    parser.add_argument('--l', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    r0 = np.random.RandomState(args.seed)
    ppo(lambda : suite.load(args.env,args.task,task_kwargs={'random':r0}), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs)