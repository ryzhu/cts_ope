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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor


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
            num_switches = 0

            while not env.is_done(episode_length=episode_length):
            #     action = self.policy(state, eps)
                state_list.append(state)
                prob_take_action_1 = policy(state, prev_action)
                
                action = 3 * int(np.random.random() < prob_take_action_1)
                action_list.append(action)
                num_switches += int(int(action != 0) != prev_action)
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
            # ep_reward = np.mean(rewards)
            ep_reward = reward # outcome is final reward
            data.append({"states": np.array(state_list), "actions": np.array(action_list), "outcome": ep_reward,
                "num_switches": num_switches})
                #    "policy_probs": policy_probs})
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

### AIPW helper functions ###
def get_switch_model(obs_data_train):
    states = np.vstack([traj["states"] for traj in obs_data_train])
    actions = np.hstack([traj["actions"] for traj in obs_data_train])
    prev_actions = np.concatenate([[0], actions[:-1]])
    switch = (prev_actions != actions).astype(int)
    switch_model = LogisticRegression(max_iter=10000).fit(states, switch)
    return switch_model

def pihat_obs_helper(obs, prev_action, switch_model):
    pred_switch_probs = switch_model.predict_proba(obs)[:, 1]
    switch_probs_01 = pred_switch_probs # P(1 | 0)
    stay_probs_11 = 1 - pred_switch_probs # P(1 | 1)
    return np.where(prev_action == 0, switch_probs_01, stay_probs_11)

def get_Q_model(obs_data_train, switch_model, eval_pol):
    pihat_obs = lambda obs, prev_action: pihat_obs_helper(obs, prev_action, switch_model)
    weighted_outcomes = []
    SA = []
    for traj in obs_data_train:
        states = traj["states"]
        actions = traj["actions"]
        weights = policy_prob_traj(
            eval_pol, states, actions) / policy_prob_traj(pihat_obs, states, actions)
        prod_weights = np.cumprod(weights[::-1])[::-1]
        weighted_outcomes.append(prod_weights * traj["outcome"])
        SA.append(np.hstack([states, np.arange(0, total_days, dt).reshape(-1, 1), actions.reshape(-1, 1)]))
    weighted_outcomes = np.hstack(weighted_outcomes)
#     states = np.vstack([np.hstack(
#         [traj["states"], np.arange(0, total_days, dt).reshape(-1, 1)]) 
#                         for traj in obs_data_train])
#     actions = np.hstack([traj["actions"] for traj in obs_data_train])
#     SA = np.hstack([states, actions.reshape(-1, 1)])
    SA = np.vstack(SA)
    Q_hat = RandomForestRegressor(max_depth=2, random_state=0).fit(SA, weighted_outcomes)
    return Q_hat

def get_aipw_helper(traj, switch_model, Q_hat, eval_pol):
    """ Return a single IPW and AIPW est using a single trajectory. """
    pihat_obs = lambda obs, prev_action: pihat_obs_helper(obs, prev_action, switch_model)
    # def parallel_helper(traj):
    states = traj["states"]
    actions = traj["actions"]
    prev_actions = np.concatenate([[0], actions[:-1]])
    t = np.arange(0, total_days, dt).reshape(-1, 1)
    sa = np.hstack([states, t, actions.reshape(-1, 1)])
    sa0 = np.hstack([states, t, np.zeros(t.shape)])
    sa1 = np.hstack([states, t, np.ones(t.shape)])

    # compute mean fn ests
    Q_hat_sa = Q_hat.predict(sa)
    Q_hat_sa0 = Q_hat.predict(sa0)
    Q_hat_sa1 = Q_hat.predict(sa1)

    pi_eval_1 = threshold_eval_pol(states, prev_actions)
    pi_eval_0 = 1 - threshold_eval_pol(states, prev_actions)
    V_hat_s = pi_eval_0 * Q_hat_sa0 + pi_eval_1 * Q_hat_sa1

    # compute ip weights
    weights = policy_prob_traj(
        eval_pol, states, actions) / policy_prob_traj(pihat_obs, states, actions)
    prod_weights = np.cumprod(weights[::-1])[::-1]

    weighted_Q_sa = prod_weights * Q_hat_sa
    weighted_V_s = np.concatenate([[1], prod_weights[:-1]]) * V_hat_s
    control_variates = weighted_V_s - weighted_Q_sa

    ipw_est = traj["outcome"]*np.prod(weights)
    aipw_est = ipw_est + np.sum(control_variates)
    return ipw_est, aipw_est         

def get_aipw_evals(obs_data_eval, switch_model, Q_hat, eval_pol):
    aipw_ests = []
    ipw_ests = []
    with mp.Pool(mp.cpu_count()) as pool:
        res = pool.starmap_async(get_aipw_helper, [[traj, switch_model, Q_hat, eval_pol] for traj in obs_data_eval]) 
        ests = res.get()
    ipw_ests = [est[0] for est in ests]
    aipw_ests = [est[0] for est in ests]

        # for ests in tqdm(pool.imap_unordered(get_aipw_helper, 
        # [[traj] for traj in obs_data_eval])):
        #     ipw_ests.append(ests[0])
        #     aipw_ests.append(ests[1])
    return ipw_ests, aipw_ests    

def AIPW_eval(obs_data, eval_pol):
    """ Get AIPW ests. """
    # Split data
    num_obs = len(obs_data)
    obs_data_1 = obs_data[:num_obs//2]
    obs_data_2 = obs_data[num_obs//2:]
    
    switch_model_1 = get_switch_model(obs_data_1)
    switch_model_2 = get_switch_model(obs_data_2)
    
    # def pihat_obs_1(obs, prev_action):
    #     """Pihat trained on split 1. """
    #     return pihat_obs_helper(obs, prev_action, switch_model_1)
    # def pihat_obs_2(obs, prev_action):
    #     """Pihat trained on split 2. """
    #     return pihat_obs_helper(obs, prev_action, switch_model_2)
    
    Q_hat_1 = get_Q_model(obs_data_1, switch_model_1, eval_pol) # Qhat trained on split 1
    Q_hat_2 = get_Q_model(obs_data_2, switch_model_2, eval_pol) # Qhat trained on split 2
    
    ### Cross evaluation ###
    
    ipw_ests_1, aipw_ests_1 = get_aipw_evals(obs_data_2, switch_model_1, Q_hat_1, eval_pol) # train on split 1, eval on split 2
    ipw_ests_2, aipw_ests_2 = get_aipw_evals(obs_data_1, switch_model_2, Q_hat_2, eval_pol) # train on split 2, eval on split 1
    
    ipw_ests = ipw_ests_1 + ipw_ests_2
    aipw_ests = aipw_ests_1 + aipw_ests_2
    return ipw_ests, aipw_ests

if __name__ == '__main__':  # <- prevent RuntimeError for 'spawn'
    # and 'forkserver' start_methods
    total_days = 60

    ##### Get monte carlo policy rollouts. #####
    num_monte_carlo_rollouts = int(1e4)
    outcomes = {}
    # for V_weight in [-1, -2, -3]:
    #     for E_weight in [1, 2, 3]:
    #         for c in [-1, 0, 1]:
    V_weight, E_weight, c = -2, 2, 0
    param_string = "Vw: {}, Ew: {}, c: {}".format(V_weight, E_weight, c)
    print(param_string)

    B = 0.2
    policy_type = "thresh"
    # for dt in [0.1, 0.3, 1, 3, 10]:
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
    #     with open('results/monte_carlo_{}_eval_T_{}_dt_{}_B_{}.pickle'.format(policy_type, total_days, dt, B), 'wb') as f:
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
    # for dt in [0.1, 0.3, 1, 3]:
    #     def log_obs_pol(obs, prev_action):
    #         return log_linear_policy(
    #             obs, prev_action, np.array([0, 0, 0, 0, V_weight, E_weight]), c, B, dt, raw_state=False)
    #     def get_obs_data(null_arg):
    #         return get_data(log_obs_pol, dt, total_days, 1)
    #     results = []
    #     with mp.Pool(mp.cpu_count()) as pool:
    #         for traj in tqdm(pool.imap_unordered(get_obs_data, [0 for _ in range(num_obs_trajs)])):
    #             results.extend(traj)
    #     with open('results/obs_log_T_{}_dt_{}_B_{}_n_{}.pickle'.format(total_days, dt, B, num_obs_trajs), 'wb') as f:
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
    num_seeds = 1
    num_obs_trajs_list = [int(1e4)] # [int(3e2), int(1e3), int(3e3)] #, int(1e4), int(1e5)]
    B_obs, B_eval = 0.1, 0.1
    for num_obs_trajs in tqdm(num_obs_trajs_list):
        for dt in tqdm([0.3, 1, 3]):
            def log_obs_pol(obs, prev_action):
                return log_linear_policy(
                    obs, prev_action, np.array([0, 0, 0, 0, V_weight, E_weight]), c, B_obs, dt, raw_state=False)
            def get_obs_data(null_arg):
                return get_data(log_obs_pol, dt, total_days, 1)
            # def log_eval_pol(obs, prev_action):
            #     return log_linear_policy(
            #         obs, prev_action, np.array([0, 0, 0, 0, V_weight, E_weight]), c, B_eval, dt, raw_state=False)
            def threshold_eval_pol(obs, prev_action):
                return constant_threshold_policy(
                obs, prev_action, np.array([0, 0, 0, 0, V_weight, E_weight]), c, B, dt, raw_state=False)
            
            # all_ests = []
            all_IPW_ests = []
            all_AIPW_ests = []
            for _ in tqdm(range(num_seeds)):
                obs_data = []
                with mp.Pool(mp.cpu_count()) as pool:
                    for traj in tqdm(pool.imap_unordered(get_obs_data, [0 for _ in range(num_obs_trajs)])):
                        obs_data.extend(traj)
            

                # IPW_ests, IPW_weights = IPW_eval(results, log_obs_pol, threshold_eval_pol)
                IPW_ests, AIPW_ests = AIPW_eval(obs_data, threshold_eval_pol)
                all_IPW_ests.append([np.mean(IPW_ests), stats.sem(IPW_ests)])
                all_AIPW_ests.append([np.mean(AIPW_ests), stats.sem(AIPW_ests)])
            eval_data = {"Vw_eval: {}, Ew_eval: {}, c_eval: {}, B_eval: {}".format(
                V_weight, E_weight, c, B_eval): {"IPW": all_IPW_ests, "AIPW": all_AIPW_ests}}

            with open('results/aipw_T_{}_dt_{}_Bobs_{}_n_{}.pickle'.format(total_days, dt, B_obs, num_obs_trajs), 'wb') as f:
                pickle.dump(eval_data, f)