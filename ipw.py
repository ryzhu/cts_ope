import sys
print(sys.path)
import numpy as np
# with dview.sync_imports():
#     from numpy import exp, where, mean, minimum
from hiv import HIVTreatment as model
import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())
pool = mp.Pool(mp.cpu_count())


import pickle
# import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import sem
from scipy import stats


### Environment Params ###
# dt = 1
# episode_length = 1000/dt
perturb_rate = 0.0
num_actions = 4
num_states = 6
env_params = {'p_k2': -0.03921113587893166,
 'p_k1': 0.20856792856393241,
 'p_f': 0.18057095024380307,
 'p_lambda2': 0.46816823561605014,
 'p_lambda1': -0.1946112184997162,
 'p_bE': -0.046679126072529116,
 'p_Kd': -0.09565291521906652,
 'p_Kb': -0.38570726203357936,
 'p_m1': 0.313893964746566,
 'p_m2': -0.024093381660099132,
 'p_d_E': -0.03485294142562121,
 'p_lambdaE': -0.11146573677164344}

## Rollout a Trajectory ##
def encode_action(action):
    a = np.zeros(num_actions)
    a[action] = 1
    return a

def random_policy(state):
    return np.random.randint(0, num_actions)

def constant_hazard_threshold_policy(obs, prev_action, beta, c, B, dt):
    """ Prev_action is binary 0-1. Returns probability of taking action 1. """
    if prev_action==0 and (10**obs)@beta >= c:
        return B*dt
    elif prev_action != 0 and (10**obs)@beta < c:
        return 1 - B*dt
    return int(prev_action != 0)

def log_linear_policy(obs, prev_action, beta, c, B, dt):
    """ Prev_action is binary 0-1. Returns P(taking action 1) = exp(beta@obs + c) dt. 
        Setup is in prep for logistic regression down the line. """
    switch_probs_01 = np.minimum(B * dt / (1 + np.exp((10**obs) @ beta - c)), 1) # P(1 | 0)
    stay_probs_11 = 1 - np.minimum(B * dt / (1 + np.exp(- ((10**obs) @ beta - c))), 1) # P(1 | 1)
    return np.where(prev_action == 0, switch_probs_01, stay_probs_11)
    
def get_data_parallel(policy, dt=5, total_days=1000, num_patients=30):
#     track = True

    
#     data = []

#     for _ in tqdm(range(num_patients)): # parallelize this
    def run_traj(seed):
            import sys
            sys.path.append('/Users/henryzhu/Research/data/RepBM')
            from hiv_domain.hiv_simulator.hiv import HIVTreatment as model
            import numpy as np
            episode_length = total_days/dt
            env = model(perturb_rate = perturb_rate, dt=dt)

            np.random.seed(seed)

            env.reset(perturb_params =  True, **env_params) # default to params from Liu
            state = env.observe()   
            # task is done after max_task_examples timesteps or when the agent enters a terminal state
    #         ep_list = []
            state_list = []
            action_list = []
            ep_reward = 0
            prev_action = 0
            rewards = []
            policy_probs = []

            while not env.is_done(episode_length=episode_length):
            #     action = self.policy(state, eps)
                state_list.append(state)
                prob_take_action_1 = policy(state, prev_action)
                
                action = 3 * int(np.random.random() < prob_take_action_1)
                action_list.append(action)
                reward, next_state = env.perform_action(action, perturb_params=True, **env_params)
    #             if not track: tmp.append(np.hstack([state,action,reward, next_state]))
    #             else: 

    #             ep_list.append(np.array([state, encode_action(action),reward,next_state, ins]))
                state = next_state
                prev_action = int(action != 0)
            #     ep_reward += (reward*self.gamma**self.task.t)
#                 ep_reward += reward
                rewards.append(reward)
                if action == 0:
                    policy_probs.append(1 - prob_take_action_1)
                else:
                    policy_probs.append(prob_take_action_1)
    #         if track:
    #             pass
                # print(np.unique(action_list, return_counts = True),ep_reward)
#             ep_reward /= episode_length
            ep_reward = np.median(rewards)
            return {"states": np.array(state_list), "actions": np.array(action_list), "outcome": ep_reward,
                   "policy_probs": policy_probs}
#             data.append()
#     with ipp.Cluster() as rc:
        
#         # get a view on the cluster
#         view = rc.load_balanced_view()
        
#         rc[:].push(dict(
#             log_linear_policy=log_linear_policy,
#             beta_1=beta_1,
#             c1=c1
#         ))a
        # submit the tasks
    dview.push(dict(env_params=preset_hidden_params[ins],
                   perturb_rate=perturb_rate,
                   total_days=total_days,
                   dt=dt,
                   policy=policy))
    asyncresult = dview.map_async(run_traj, list(range(num_patients)))
    # wait interactively for results
    asyncresult.wait_interactive()
#     asyncresult.wait()
    # retrieve actual results
    data = asyncresult.get()
    return data

def get_data(policy, dt=5, total_days=1000, num_patients=30):
    data = []
    for _ in tqdm(range(num_patients)): # parallelize this
            episode_length = total_days/dt
            env = model(perturb_rate = perturb_rate, dt=dt)
            env.reset(perturb_params =  True, **env_params) # default to params from Liu
            state = env.observe()   
            # task is done after max_task_examples timesteps or when the agent enters a terminal state
    #         ep_list = []
            state_list = []
            action_list = []
            ep_reward = 0
            prev_action = 0
            rewards = []
            policy_probs = []

            while not env.is_done(episode_length=episode_length):
            #     action = self.policy(state, eps)
                state_list.append(state)
                prob_take_action_1 = policy(state, prev_action)
                
                action = 3 * int(np.random.random() < prob_take_action_1)
                action_list.append(action)
                reward, next_state = env.perform_action(action, perturb_params=True, **env_params)
    #             if not track: tmp.append(np.hstack([state,action,reward, next_state]))
    #             else: 

    #             ep_list.append(np.array([state, encode_action(action),reward,next_state, ins]))
                state = next_state
                prev_action = int(action != 0)
            #     ep_reward += (reward*self.gamma**self.task.t)
#                 ep_reward += reward
                rewards.append(reward)
                if action == 0:
                    policy_probs.append(1 - prob_take_action_1)
                else:
                    policy_probs.append(prob_take_action_1)
            ep_reward = np.median(rewards)
            data.append({"states": np.array(state_list), "actions": np.array(action_list), "outcome": ep_reward,
                   "policy_probs": policy_probs})
    return data

dt=5
B=0.1
total_days=1000
beta_1, c1 = np.array([0, 0, 0, 0, 0.00002, -0.2]), -3
loglin_pol_1 = lambda obs, prev_action: log_linear_policy(
    obs, prev_action, beta_1, c1, B, dt)

results = []
def collect_result(result):
    global results
    results.append(result)

for traj in tqdm(pool.istarmap_unordered(get_data, [[loglin_pol_1, dt, total_days, 1] for _ in range(int(1e4))])):
    results.append(traj)
# for i, row in enumerate(data):
#     pool.apply_async(get_data, args=(i, row, 4, 8), callback=collect_result)


# data_loglin_pol_1_dt_5 = get_data(loglin_pol_1, dt=5, )

with open('results/loglin_dt_5_B_01.pickle', 'wb') as f:
    pickle.dump(results, f)