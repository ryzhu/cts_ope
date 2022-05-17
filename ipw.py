import sys
print(sys.path)
import numpy as np
# with dview.sync_imports():
#     from numpy import exp, where, mean, minimum
from hiv import HIVTreatment as model
import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())


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
    np.random.seed()
    data = []
    for _ in range(num_patients): # parallelize this
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
            # ep_reward = np.median(rewards)
            ep_reward = np.mean(rewards)
            data.append({"states": np.array(state_list), "actions": np.array(action_list), "outcome": ep_reward,
                   "policy_probs": policy_probs})
    return data

### Define observational policies, which have a logistic structure. ###
def random_policy(state):
    return np.random.randint(0, num_actions)

# def constant_hazard_threshold_policy(obs, prev_action, beta, c, B, dt):
#     """ Prev_action is binary 0-1. Returns probability of taking action 1. """
#     if prev_action==0 and (10**obs)@beta >= c:
#         return B*dt
#     elif prev_action != 0 and (10**obs)@beta < c:
#         return 1 - B*dt
#     return int(prev_action != 0)

def log_linear_policy(obs, prev_action, beta, c, B, dt, raw_state=False):
    """ Prev_action is binary 0-1. Returns P(taking action 1) = exp(beta@obs + c) dt. 
        Setup is in prep for logistic regression down the line. """
    if raw_state:
        obs = 10**obs
    switch_probs_01 = np.minimum(B * dt / (1 + np.exp(obs @ beta - c)), 1) # P(1 | 0)
    stay_probs_11 = 1 - np.minimum(B * dt / (1 + np.exp(- (obs @ beta - c))), 1) # P(1 | 1)
    return np.where(prev_action == 0, switch_probs_01, stay_probs_11)

def policy_prob_traj(policy, obs, actions):
    """ Returns the probability of taking the trajectory under the policy. 
        Obs - horizon x state dim 2D array, Actions - 1D array of length horizon """
    prev_actions = np.concatenate([[0], actions[:-1]])
    pi_1 = policy(obs, prev_actions) # P(A_t = 1 | X_t, A_t-1)
    pi_0 = 1 - pi_1 # P(A_t = 0 | X_t, A_t-1)
    return np.where(actions == 0, pi_0, pi_1)

def IPW_eval(obs_data, pi_obs, pi_eval):
    IPW_weighted_vals = []
    for traj in obs_data:
        # Use log trick to avoid overflow
        log_prob_traj_o = np.sum(np.log(policy_prob_traj(pi_obs, traj["states"], traj["actions"])))
        log_prob_traj_e = np.sum(np.log(policy_prob_traj(pi_eval, traj["states"], traj["actions"])))
        ipw = np.exp(log_prob_traj_e - log_prob_traj_o)
        IPW_weighted_vals.append(ipw * traj["outcome"])
    return np.mean(IPW_weighted_vals), stats.sem(IPW_weighted_vals)

# dt=5
# B=0.1
# total_days=1000
# beta_1, c1 = np.array([0, 0, 0, 0, 0.00002, -0.2]), -3
# def loglin_pol_1(obs, prev_action):
#     return log_linear_policy(
#     obs, prev_action, beta_1, c1, B, dt, raw_state=True)
# # loglin_pol_1 = lambda obs, prev_action: log_linear_policy(
# #     obs, prev_action, beta_1, c1, B, dt)

# beta_2, c2 = np.array([0, 0, 0, 0, -2, 2]), 0
# def loglin_pol_2(obs, prev_action):
#     return log_linear_policy(
#     obs, prev_action, beta_2, c2, B, dt, raw_state=False)

# results = []
# def collect_result(result):
#     global results
#     results.append(result)

# def get_data_dt_5(policy):
#     return get_data(policy, 5, total_days, 1)

### Define evaluation threshold policies. ###
def constant_threshold_policy(obs, prev_action, beta, c, B, dt, raw_state=False):
    """ Prev_action is binary 0-1. Returns probability of taking action 1. """
    if raw_state:
        obs = 10**obs

    switch_probs_01 = B * dt * (obs @ beta - c <= 0) # P(1 | 0)
    stay_probs_11 = 1 - B * dt * (obs @ beta - c >= 0) # P(1 | 1)
    return np.where(prev_action == 0, switch_probs_01, stay_probs_11)

##### Define IPW ests #####
def policy_prob_traj(policy, obs, actions):
    """ Returns the probability of taking the trajectory under the policy. 
        Obs - horizon x state dim 2D array, Actions - 1D array of length horizon """
    prev_actions = np.concatenate([[0], actions[:-1]])
    pi_1 = policy(obs, prev_actions) # P(A_t = 1 | X_t, A_t-1)
    pi_0 = 1 - pi_1 # P(A_t = 0 | X_t, A_t-1)
    return np.where(actions == 0, pi_0, pi_1)

def IPW_eval(obs_data, pi_obs, pi_eval):
    IPW_weighted_vals = []
    ip_weights = []
    for traj in obs_data:
        # Use log trick to avoid overflow
        log_prob_traj_o = np.sum(np.log(policy_prob_traj(pi_obs, traj["states"], traj["actions"])))
        log_prob_traj_e = np.sum(np.log(policy_prob_traj(pi_eval, traj["states"], traj["actions"])))
        ipw = np.exp(log_prob_traj_e - log_prob_traj_o)
        IPW_weighted_vals.append(ipw * traj["outcome"])
        ip_weights.append(ipw)
    return IPW_weighted_vals, ip_weights

if __name__ == '__main__':  # <- prevent RuntimeError for 'spawn'
    # and 'forkserver' start_methods
    total_days = 1000

    ##### Get monte carlo policy rollouts. #####
    num_monte_carlo_rollouts = int(1e3)
    outcomes = {}
    # for V_weight in [-1, -2, -3]:
    #     for E_weight in [1, 2, 3]:
    #         for c in [-1, 0, 1]:
    V_weight, E_weight, c = -2, 2, 0
    param_string = "Vw: {}, Ew: {}, c: {}".format(V_weight, E_weight, c)
    print(param_string)

    # B = 0.3
    # policy_type = "log"
    # for dt in [0.1]: #, 0.3, 1, 3]:
    #     def threshold_eval_pol(obs, prev_action):
    #         return constant_threshold_policy(
    #         obs, prev_action, np.array([0, 0, 0, 0, V_weight, E_weight]), c, B, dt, raw_state=False)
    #     def log_obs_pol_eval(obs, prev_action):
    #         return log_linear_policy(
    #             obs, prev_action, np.array([0, 0, 0, 0, V_weight, E_weight]), c, B, dt, raw_state=False)
    #     def get_monte_carlo_eval_data(null_arg):
    #         if policy_type == "thresh":
    #             return get_data(threshold_eval_pol, dt, total_days, 1)
    #         elif policy_type == "log":
    #             return get_data(log_obs_pol_eval, dt, total_days, 1)
        
    #     trajs = []
    #     with mp.Pool(mp.cpu_count()) as pool:
    #         for traj in tqdm(pool.imap_unordered(get_monte_carlo_eval_data, [0 for _ in range(num_monte_carlo_rollouts)])):
    #             trajs.extend(traj)
    #     outcomes[param_string] = np.array([traj["outcome"] for traj in trajs])
    #     with open('results/monte_carlo_{}_eval_dt_{}_B_{}.pickle'.format(policy_type, dt, B), 'wb') as f:
    #         pickle.dump(outcomes, f)
    
    # dt = 0.1
    # for B in [0.99, 1, 1.01]:
    #     def threshold_eval_pol(obs, prev_action):
    #         return constant_threshold_policy(
    #         obs, prev_action, np.array([0, 0, 0, 0, V_weight, E_weight]), c, B, dt, raw_state=False)
    #     def get_monte_carlo_eval_data(null_arg):
    #         return get_data(threshold_eval_pol, dt, total_days, 1)
        
    #     trajs = []
    #     with mp.Pool(mp.cpu_count()) as pool:
    #         for traj in tqdm(pool.imap_unordered(get_monte_carlo_eval_data, [0 for _ in range(num_monte_carlo_rollouts)])):
    #             trajs.extend(traj)
    #     outcomes[param_string] = np.array([traj["outcome"] for traj in trajs])
    #     with open('results/monte_carlo_thresh_eval_dt_{}_B_{}.pickle'.format(dt, B), 'wb') as f:
    #         pickle.dump(outcomes, f)

    ##### Get obs data. #####
    # num_obs_trajs = int(1e3)
    # B = 0.2
    # for dt in [0.3, 1, 3]:
    #     def log_obs_pol(obs, prev_action):
    #         return log_linear_policy(
    #             obs, prev_action, np.array([0, 0, 0, 0, V_weight, E_weight]), c, B, dt, raw_state=False)
    #     def get_obs_data(null_arg):
    #         return get_data(log_obs_pol, dt, total_days, 1)
    #     results = []
    #     with mp.Pool(mp.cpu_count()) as pool:
    #         for traj in tqdm(pool.imap_unordered(get_obs_data, [0 for _ in range(num_obs_trajs)])):
    #             results.extend(traj)
    #     with open('results/obs_log_dt_{}_B_{}_n_{}.pickle'.format(dt, B, num_obs_trajs), 'wb') as f:
    #         pickle.dump(results, f)

    # dt = 0.1
    # for B in [0.99, 1, 1.01]:
    #     def log_obs_pol(obs, prev_action):
    #         return log_linear_policy(
    #             obs, prev_action, np.array([0, 0, 0, 0, V_weight, E_weight]), c, B, dt, raw_state=False)
    #     def get_obs_data(null_arg):
    #         return get_data(log_obs_pol, dt, total_days, 1)
    #     results = []
    #     with mp.Pool(mp.cpu_count()) as pool:
    #         for traj in tqdm(pool.imap_unordered(get_obs_data, [0 for _ in range(num_obs_trajs)])):
    #             results.extend(traj)
    #     with open('results/obs_log_dt_{}_B_{}_n_{}.pickle'.format(dt, B, num_obs_trajs), 'wb') as f:
    #         pickle.dump(results, f)


    ##### Get IPW ests. #####
    num_obs_trajs = int(3e3)
    B_obs, B_eval = 0.2, 0.3
    for dt in [0.1, 0.3, 1, 3]:
        def log_obs_pol(obs, prev_action):
            return log_linear_policy(
                obs, prev_action, np.array([0, 0, 0, 0, V_weight, E_weight]), c, B_obs, dt, raw_state=False)
        def get_obs_data(null_arg):
            return get_data(log_obs_pol, dt, total_days, 1)
        results = []
        with mp.Pool(mp.cpu_count()) as pool:
            for traj in tqdm(pool.imap_unordered(get_obs_data, [0 for _ in range(num_obs_trajs)])):
                results.extend(traj)
        
        def log_eval_pol(obs, prev_action):
            return log_linear_policy(
                obs, prev_action, np.array([0, 0, 0, 0, V_weight, E_weight]), c, B_eval, dt, raw_state=False)
        IPW_ests, IPW_weights = IPW_eval(results, log_obs_pol, log_eval_pol)
        IPW_data = {"Vw_eval: {}, Ew_eval: {}, c_eval: {}, B_eval: {}".format(
            V_weight, E_weight, c, B_eval): {"IPW ests": IPW_ests, "IPW_weights": IPW_weights}}

        with open('results/ipw_dt_{}_Bobs_{}_n_{}.pickle'.format(dt, B_obs, num_obs_trajs), 'wb') as f:
            pickle.dump(IPW_data, f)