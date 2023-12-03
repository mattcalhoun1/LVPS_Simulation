# Utilities borrowed from Dr. Bean at Maryville University
class SB3Agent:
    def __init__(self, model, seed=None, deterministic=True, normalizer=None): 
        self.model = model
        self.seed = seed
        self.deterministic = deterministic
        self.normalizer = normalizer
    def select_action(self, state):
        if self.seed is not None:
            from stable_baselines3.common.utils import set_random_seed
            set_random_seed(self.seed)
        if self.normalizer is not None:
            state = self.normalizer(state)
        a = self.model.predict(state, deterministic=self.deterministic)[0]
        return a

def state_str(env, state):
    '''
    Represents states as strings for the sake of displaying them in reports. 
    '''
    import numpy as np
    np.set_printoptions(suppress=True)
    if env.spec.id == 'CartPole-v1':
        ss = str(state.round(3))
    else:
        ss = str(state)
    return ss

def encode_state(state):
    '''
    Encodes states for use as keys in dictionaries. 
    -- Lists --> Tuples
    -- Arrays --> Bytes
    -- Integers/Floats/Strings --> Unaffected
    '''
    import numpy as np
    if type(state) == list: return tuple(state)
    if isinstance(state, np.ndarray): return state.tobytes()
    return state

def decode_state(env, state):
    import numpy as np
    if env.spec.id == 'FrozenLake-v1':
        return int(state)
    

def set_seed(seed):
    import numpy as np
    if seed is None:
        return None
    np_state = np.random.get_state()
    np.random.seed(seed)
    return np_state

def unset_seed(np_state):
    import numpy as np
    if np_state is None:
        return None
    np.random.set_state(np_state)


def generate_episode(
    env, agent,  max_steps=None, init_state=None, random_init_action=False, 
    epsilon=0.0, seed=None, verbose=False, atari=False):
    
    import numpy as np
    import time
    
    #--------------------------------------------------------
    # Set seeds
    #--------------------------------------------------------
    np_state = set_seed(seed)
    
    
    #-----------------------------------------------------------------------
    # Check to see if environment is framestacked
    #-----------------------------------------------------------------------
    frame_stacked = True if 'VecFrameStack' in str(type(env)) else False
    
    #--------------------------------------------------------
    # Reset Environment
    #--------------------------------------------------------
    if frame_stacked:
        # Reset the base environment, providing a seed
        if seed is not None:
            env.unwrapped.envs[0].unwrapped.reset(seed=int(seed))  
            env.action_space.seed(int(seed))
        else:
            env.unwrapped.envs[0].unwrapped.reset()   
        # Reset vec_env
        state = env.reset()
        
    else:
        if seed is not None:
            state, info = env.reset(seed=int(seed))
            env.action_space.seed(int(seed))
        else:
            state, info = env.reset()
    
    '''
    #--------------------------------------------------------
    # Reset Environment
    #--------------------------------------------------------
    if atari:
        # Reset the base environment, providing a seed
        if seed is not None:
            env.unwrapped.envs[0].unwrapped.reset(seed=int(seed))  
            env.action_space.seed(int(seed))
        else:
            env.unwrapped.envs[0].unwrapped.reset()  
        # Reset vec_env
        state = env.reset()
    
    else:
        if seed is not None:
            state, info = env.reset(seed=int(seed))
            env.action_space.seed(int(seed))
        else:
            state, info = env.reset(seed=int(seed))
    '''
                        
    #--------------------------------------------------------
    # Set initial state (Used for exploring starts)
    #--------------------------------------------------------
    if init_state is not None:
        env.unwrapped.s = init_state
        state = init_state
    
    # In case init state was not specified, store it for later.
    init_state = state

    #--------------------------------------------------------
    # Lists to store information
    #--------------------------------------------------------
    s_list, a_list, r_list, d_list, c_list, i_list =\
        [], [], [], [], [], []
    
    #--------------------------------------------------------
    # Loop for max steps episodes
    #--------------------------------------------------------
    t = 0
    lives = None            # Used to track when life lost for Atari
    new_lives = None        # Used to track when life lost for Atari
    if max_steps is None:
        max_steps = float('inf')
    while t < max_steps:
        t += 1
        
        #--------------------------------------------------------
        # Determine if action should be selected at random. 
        # True if using exp starts and t==1, or if roll < epsilon
        # For sake of efficiiency, don't roll unless needed.
        #--------------------------------------------------------
        random_action = False
        if random_init_action and t == 1:
            random_action = True
        if epsilon > 0:
            roll = np.random.uniform(0,1)
            if roll < epsilon:
                random_action = True
        
        #--------------------------------------------------------
        # Select action
        #--------------------------------------------------------
        if random_action:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)
        
        #--------------------------------------------------------
        # Check to see if reset is needed for Atari Environment
        # Required when a life is lost
        #--------------------------------------------------------
        if frame_stacked:
            if t == 2:              
                lives = new_lives   # Both start as None
            if lives != new_lives:
                action = 1                
            lives = new_lives
        
        
        #--------------------------------------------------------
        # Apply action
        #--------------------------------------------------------
        if frame_stacked:
            # SB3 models will retun the action in a list already
            # But random agents will not. 
            if not isinstance(action, np.ndarray) and not isinstance(action, list):
                action = [action]
            state, reward, done, info = env.step(action)
            reward = reward[0]
            done = done[0]
            truncated = False
            new_lives = info[0]['lives']
        else:
            state, reward, done, truncated, info = env.step(action)
        
        a_list.append(action)
        s_list.append(state)
        r_list.append(reward)
        d_list.append(done)
        c_list.append(truncated)
        i_list.append(info)
        
        if done:
            break

    if verbose:
        ss_list = [state_str(env, s) for s in s_list] # format_states
        i_list = [str(i) for i in i_list]
        d_list = [str(t) for t in d_list]
        
        wn = 6
        wa = max([len(str(a)) for a in a_list]) + 2
        wa = max(wa, 8)
        ws = max([len(s) for s in ss_list]) + 2
        ws = max(ws, 11)
        wr = 8
        wt = 12
        wi = max([len(i) for i in i_list])
        wi = max(wi, 4)
        w = wn + wa + ws + wr + wt + wi   
        
        #--------------------------------------------------------
        # Output Header
        #--------------------------------------------------------   
    
        print(f'{"step":<{wn}}{"action":<{wa}}{"new_state":<{ws}}' +
              f'{"reward":<{wr}}{"terminated":<{wt}}{"info":<{wi}}')
        print('-'*w)
        
        for n, (a, s, r, d, i) in enumerate(
            zip(a_list, ss_list, r_list, d_list, i_list)):
     
            print(f'{n:<{wn}}{a:<{wa}}{s:<{ws}}' + 
                  f'{r:<{wr}}{d:<{wt}}{i:<{wi}}')
        print('-'*w)
        print(t, 'steps completed.')
        
    #--------------------------------------------------------
    # Create history dictionary 
    # Note: States will be one longer than the others.
    #--------------------------------------------------------
    history = {
        'states' : [init_state] + s_list,
        'actions' : a_list,
        'rewards' : r_list 
    } 
    
    #------------------------------------------------------------
    # Unset the seed
    #------------------------------------------------------------
    unset_seed(np_state)
       
    return history
        

def evaluate(env, agent, gamma, episodes, max_steps=1000, seed=None, 
             check_success=False, show_report=True, atari=False):
    import numpy as np
    
    np_state = set_seed(seed)
    
    returns = []
    lengths = []
    success = []
    num_success = 0
    num_failure = 0
    len_success = 0
    len_failure = 0
    
    for n in range(episodes):
        ep_seed = np.random.choice(10**6)
        history = generate_episode(
            env=env, agent=agent, max_steps=max_steps, epsilon=0.0, 
            seed=ep_seed, verbose=False, atari=atari
        )
        
        #------------------------------------------------------------
        # Calcuate return at initial state
        #------------------------------------------------------------
        num_steps = len(history['actions'])
        G0 = 0
        for t in reversed(range(num_steps)):
            G0 = history['rewards'][t] + gamma * G0
        
        returns.append(G0)
        lengths.append(num_steps)
        
        
        #------------------------------------------------------------
        # Update success rate info, if requested
        #------------------------------------------------------------
        if check_success:
            success.append(True if env.status == 'success' else False)
            if env.status == 'success':
                num_success += 1
                len_success += len(history['actions'])
            else:
                num_failure += 1
                len_failure += len(history['actions'])
    
    #------------------------------------------------------------
    # Build stats report
    #------------------------------------------------------------
    
    stats = {
        'mean_return' : np.mean(returns),
        'stdev_return' : np.std(returns),
        'mean_length' : np.mean(lengths),
        'stdev_length' : np.std(lengths)
    }
    
    if check_success:
        stats.update({
            #'sr' : num_success / episodes,
            #'avg_len_s' : None if num_success == 0 else len_success / num_success,
            #'avg_len_f' : None if num_failure == 0 else len_failure / num_failure
            'sr' : np.mean(success),
            'avg_len_s' : None if num_success == 0 else len_success / num_success,
            'avg_len_f' : None if num_failure == 0 else len_failure / num_failure
         })

    if show_report:
        print(f'Mean Return:    {round(stats["mean_return"], 4)}')
        print(f'StdDev Return:  {round(stats["stdev_return"], 4)}')
        print(f'Mean Length:    {round(stats["mean_length"], 4)}')
        print(f'StdDev Length:  {round(stats["stdev_length"], 4)}')
        
        if check_success:
            print(f'Success Rate:   {round(stats["sr"], 4)}')

    unset_seed(np_state)

    return stats
