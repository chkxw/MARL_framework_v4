Please read /home/yuhao2024/miniconda3/envs/genesis/lib/python3.12/site-packages/pettingzoo/utils/env.py to understand AEC API, also, read the sample go2 VecEnv for rsl_rl lib:examples/locomotion/go2_env.py I need you to understand both MARL AEV API(../../miniconda3/envs/genesis/lib/python3.12/site-packages/pettingzoo/utils/env.py) and the VecENV interface used by rsl_rl lib (rsl_rl/rsl_rl/env/vec_env.py). 

A minor mention, the go2evn might have somproblem handling the last obs, that is always send the last obs instead of the reset obs, that's problemetic. But just keep in mind, no need to correct it for genesis, we justg need to make it correct in our env. 

Now here's the overall idea. I wanna build a MARL trianing lib based on genesis and rsl_rl, but rsl_rsl don't support MARL natively, so I need to make it think it is training single agent RL
While for the environment side, we need t make multiple robot goes in turns on their frequency. WHich nees some schedualing and carefully handle how they interact with the environemtn , so I pick the frama foundation's pettingzoo Aagent Environemnt cycle API to handle it. It would look almost as same as a normal AEC environment, just it's vecotrized, it support parellel environment natively (use n_envs from Genesis) so that all observation and action comes wikth another dimension of n_env. Also, the reset logic is more complex, we need to do autoreset for environments that the task is done. But the most tricky part is what if one robot in an environemtn is dead but the whole environmetn should still goes on, we still need to report its observation and reward to keep the vectorshape, but we need to use some way to indicate this is not valid

And, this Mother AEC environemnt will connect to serveral (number = number of robots in the MARL) VecEnv interface in order to train it. The VecEnv should not do too many thing, the only thing it need to do is to get action from the rsl_rl , report that to mother AEC in the middle of step, go the sleep, and wait for the AEC to revoke it with reward/obs/termiation/truncation returned by the last() function of the environemtn. I mean, it should not do too much things, just an interface to rsl_rl lib

Big problem, how we handle the agent die but environemtn not reset situation. I think we should just do padding alike things to indicate those are m=not valid states

Problem Solving fashion: 

- A LOT of debug logging! This is the main way we debug. If you met with problem, jsut try to add some debug looging in the place you need information, and no need to remove that afterawrds.
- If you encounter with some problem, go in to the doc folder to search for doc , also read the examples for reference, instead of blindly trying

Also, keep things in gpu, use tensor as much as possible




● Based on the AEC code, here's the detailed execution flow with function calls:

  AEC Environment Execution Loop Flow

  1. INITIALIZATION
     env = CustomAECEnv()

     env.reset(seed=None, options=None)
     └── Initializes: agents, possible_agents, terminations, truncations, 
         rewards, _cumulative_rewards, infos, agent_selection

  2. MAIN EXECUTION LOOP
     for agent in env.agent_iter(max_iter):
     │
     ├── env.agent_iter(max_iter) creates AECIterable
     │   └── AECIterable.__iter__() returns AECIterator
     │       └── AECIterator.__next__() returns env.agent_selection
     │
     ├── obs, reward, term, trunc, info = env.last()
     │   └── env.last(observe=True)
     │       ├── First calling used as return value of reset() of subVecEnv, afterwards used as return value of step() for subVecEnv (will revoke the step in subVecEnv put into waiting before). But ! if a vecEnv's corresponding agent is dead (agent terminate but environemnt still alive) DOn't send these satet information to it, it should only recieve the next reset state, so keep it waiting untill your find the agent goes back to life again
     │       ├── agent = self.agent_selection
     │       ├── observation = self.observe(agent)  # Custom implementation
     │       └── returns (observation, _cumulative_rewards[agent],
     │           terminations[agent], truncations[agent], infos[agent])
     │
     ├── if term or trunc:
     │   │   action = None
     │   │   do auto-reset in subVecEnv (Will be put the subVecEnvinto waiting for the first last() in new episode)
     │   else:
     │   │   action = policy(obs)  # Agent decides action
     │   │   The inference of policy is handled by the subVecEnv (as it is connected with rsl-rl trianing code), we don't need to do that, just pass the Execution to rsl-rl and fetch the action from the subProcEnv 's step() function 
     └── env.step(action)  # Custom implementation
         ├── Always clear the accumulative reward of current agent:self._cumulative_rewards[self.agent_selection] = 0
         ├── Always clear the immediate reward recorder env._clear_rewards()
         ├── if agent is dead:
         │   └── env._was_dead_step(action)
         │       ├── Validates action is None
         │       ├── Removes dead agent from all dicts
         │       └── Updates agent_selection to next agent (select dead agent first)
         ├── else: # Agent is alive
         │   └── Apply action (apply control to genesis, but not step environemnt yet)
         │   
         ├── Update environment state for a scheduled number of steps (till next agent's action frame, can be 0 if there another agent should act in this frame)
         ├── Calculate new immediate rewards/terminations for all
         ├── env._accumulate_rewards()
         ├── Switch to next agent (update agent_selection)
         ├── Check which env end (Currently just check if no agent is living in that env)
         │   └── If end, we need to call env.reset() to reset those pecific environment

  3. HELPER FUNCTIONS CALLED DURING EXECUTION
     ├── env._deads_step_first()  # Handle dead agents first
     ├── env._clear_rewards()     # Reset rewards dict
     ├── env._accumulate_rewards() # Add rewards to cumulative
     └── env._was_dead_step()     # Process dead agent removal

  4. TERMINATION
     Loop ends when:
     ├── No agents remain (len(env.agents) == 0)

  Key Points:
  - agent_iter() creates an iterator that yields agent_selection
  - last() gets current agent's state from previous step
  - step() processes action and advances to next agent
  - Dead agents are handled specially with _was_dead_step()
  - Rewards accumulate in _cumulative_rewards throughout the episode

● Based on the AEC code, here's all the data an AEC environment must track:

  Original AEC Core Agent Management

  - possible_agents: List of all agents that could ever appear
  - agents: List of currently active agents (subset of possible_agents)
  - agent_selection: The agent whose turn it is to act

  State Tracking (Per Agent)

  - terminations: Dict[AgentID, bool] - Whether each agent reached terminal state
  - truncations: Dict[AgentID, bool] - Whether each agent was truncated (e.g., time limit)
  - rewards: Dict[AgentID, float] - Immediate rewards usually just used as temp storage inside step()
  - _cumulative_rewards: Dict[AgentID, float] - Total rewards since agent's last action
  - infos: Dict[AgentID, dict] - Additional info for each agent

  Spaces

  - observation_spaces: Dict[AgentID, gymnasium.Space] - Observation space per agent
  - action_spaces: Dict[AgentID, gymnasium.Space] - Action space per agent

  Environment State

  - Internal environment state (custom implementation)
  - Any game/simulation-specific data

  Metadata

  - metadata: dict - Environment configuration and properties

  Optional/Helper Data

  - _skip_agent_selection: Temporary storage during dead agent handling
  - Any rendering-related state
  - Episode counters/timers (if using truncation)

  Key Design Points:
  - All per-agent data stored in dicts keyed by AgentID
  - Must maintain consistency between agents list and all dicts
  - Dead agents removed from all tracking structures
  - Rewards tracked both immediate and cumulative


  RSL_RL VecEnv Execution Loop Flow

  RSL_RL VecEnv Execution Flow (Genesis Multi-Robot Implementation)

  1. INITIALIZATION
     env = GenesisMultiRobotVecEnv(robot_configs, env_config)
     The initialization task like creating scene and robot control should all be done by the mother env
     Create as less buf as we can, since VecEnv is just an interface level thing

  2. RESET
     obs, extras = env.reset()
     Reset works of the scene should also be done all by mother env

  3. MAIN EXECUTION LOOP
     obs, rewards, dones, extras = env.step(actions)
     │
     ├── proprocess actions (for example clip)
     │
     ├── send processed action to mother env, put current process into waiting
     │
     ├── recieve termination/truncation, reward, obs from the mother AEC env
     ├── if terminate / truncate, 
     │
     └── return (obs_buf, rew_buf, reset_buf, extras)

  4. UTILITY METHODS (called as needed)
     ├── get_observations() - Get current obs without stepping
     │   ├── _update_robot_states()
     │   ├── _compute_observations()
     │   └── _update_extras()
     └── close() - Cleanup Genesis scene

  5. DATA FLOW SUMMARY
     Input:  actions (num_envs, total_action_dim)
     Output: observations (num_envs, num_obs)
            rewards (num_envs,)
            dones (num_envs,)
            extras (dict with timeout info, critic obs, etc.)

About the Vecorized feature provided by Genesis, in genesis, we can use n_envs to spawn a lot of parellel env. This is the core feature and the most important reason that we use it. However, both pettingzoo and gymnasuim is built for single env. So when interacting with the environemnt (including getting obs, setting actions), we need to do it in batch. However, Since each env is running independent experiment, each env's reset should be handled sperately. However, since we need to interact with subVecEnv, we need to keep a centuralized AgentSlector, which means, all environemtn appply action and get obs from a specific agent together, but that agent can die in some env but still alive in others

Big problems: TO make full use of the vectorized feature of genesis, we can't stricktly follows the original way to use AEC env. Now you've looked at the rsl_rl env, they both do auto reset of each env so that we just step and step to collect data. However, it is more compliucated with MARL scenario. In MARL, the robot dies != env reset, for example, if one robot crush, but the task don't fail, the other robot should still take steps to carry the episode on. However, this might only happen in some environemtn, so we cannnot tell the dead robot's subVecEnv stop collecting / sending data, as in other parelled env, this robot might still be alive. So in this case, we need some padding Observation to let the subVecEnv know this robot is dead and the subVecEnv still keep inferencing the dead robot's action to keep the tensor shape but when sending to the mother AEC env, that action should always filled with Nan. We also need to change how we handle agent dead in the AEC environment. Currently, if and agent dead, it will be removed from all dicts and list. This is kind of "hard removal", But since this agent might still be working in other envs, we actually need soft remove, that we need an n_env*number_of_robot_in_each_env to record the life and death, for those dead agent, despite it is considered dead, we will still tread it as nromal agent, that we will still call last() on it to get obs, still giving it chance to step (not removing it from all list) when it was its frame, but it it will just send nan obs/reward and recieve Nan action. So The dead state is stored in AEC, not in subVecEnv, subVecEnv knows an agent is dead by seeing the truncation/termination return from the AEC, and knows it comes back to life by seeing non-Nan observation coming

Also another problem, that I can't use any form of GUi recording. I mean, currently genesis recod by reading GLBuffer but I use virtualGL which bypass this layer. So I need to use camera entity to record images bit by bit. The sample is at dual_robot_demo.py, (this file is not your main reference ,you just look at it for the camera tracking part)

How I already have some buggy implementation vectorized_aec_env_v2.py, but it ddoesnot follow this deisgn document, but I think some compoenent you can refer to that. I think, it would be a good starting point for you

### Always use conda env named genesis 

Changes: Don't use multiprocessing for subVecEnv, ques threading for subVecEnv to avoid transfering tensor accross process

IMPORTANT CHANGE 0:
Then, apart from the basic robot Config, I wanna introduce a config bundle, which contain a list of robot config , and require all sub configs to have the SAME frequency. 
- This is not only for joint training, but work as a fundamental concept used in the whole framwork, it will be used for agent generation, and those agents will store a bundle instead of single agent
- Of course the joint training / shared network training robot config also use config bundle. For joint training, we will have a class that inherits from bundle config but without any change, just a different name. While for shared network training, it will also inherit from bundle config, but with different initializer that takes a all the other fields like a basic robot config but a list of intial position and orientation (size should match), and generate the subConfigs for each initial position and orientation accordingly.

IMPORTANT CHANGE 1:
Current request / serve fashion is too strange and not intuitive. I want the whole thing to be easier. So between the mother AEC and the subVecEnv, there are only two state, one is the subVecEnv asking for obs,rew,... from mother AEC, one is mother AEC asking for action fom subVecEnv, only these two, Which means we can use a contion variable flag to indicate the state of the subVecEnv and the mother AEC, and they both know what should do. All flags are intialized and stored in the AEC class, but each agent's class with get reference to it. And usually it's the agent's class that check and use the flag. About agent class, see CHANGE  2. Each flag is correspond to one and only one subVecEnv, each agent class may store more than one flag since each agent may be associated with more than one subVecEnv. 

At start, the all flags say AEC should send. 
For subVecEnv, all it does it wait for AEC obs,rew... -> wake up by the contion variable setting to its side -> recieve the obs,rew... -> infer action (probably done by rsl_rl on policy runner) -> send action to AEC (write into the action cache list) -> set the flag to AEC side -> wait for AEC obs,rew... -> ...
For the AEC, it will follow AEC cycle reset() -> last() -> step() -> last() -> step() -> .... But since it  server serveral robot, it need a action cache list to cache the recieved action from each agent (we already have this kind of cache in current communication manager). 
In last() it will check the flag of current selected agent, if it indicate AEC should send obs, it will send obs,rew,... and change the flag to subVecEnv side , else it will go to sleep and wait for flag. 
In step() it will check the flag of current selected agent, it will actively check the flag util it change to the AEC side (the reason why we don't actively check the action being presented in cache is that it will cause race condition with the effor to write the cache), and fetch the action from cache, apply the action in genesis simulator, and move 


New feature: joint training / shared network traing. In current setting we assume one robot will have one corresponding RL trainer, but there are common cases that:
- joint training: There are serveral robots of the SAME FREQUENCY in one scene, not even necessarily the same type, but they their obs/act get concated into one and get treated as one robot. Their rewards are the "tasK" reward that dependens on all those robots.
- shared network traing: There a reserverl robot of the SAME TYPE and SAME FREQUENCY in the scene, but they are traained with the SAME network, their obs and act have eexactlt the same dimension, and their reward are calculated in the same way. It's like, they can have physical interaction with each other as they are in the same envrionemtn, but in subVecEnv's view, each of these robot are treated as if they are in different pareellel env. If we have 10 robots in each env, and 10 parellel env, one the training side, it should see 10*10=100 parelell env in total and treat them indifferently. 

IMPORTANT CHANGE 2: 
In troduce the agent idea and corresponding class. Agent is the entity class that interact with the AEC environment, in current implementation, each robot is considered one agent. But now, I wanna change it to be each bundle of robot is one agent. Inside agent, it will remember which obs should goes to which subVecEnv and how, does it need to concat obs for each environment (if the subVec is for joint training), or it need to concat m robot each with n envs to m*n envs with the same robot (if the subVec is for shared network training). Most importantly, it will also store a reference to state flags (thoses indicating who should act, subVecEnv or AEC) to make it easier to check and set them when the agent is taking step() / calling last()

The agent have method to
- setup robot entities in a genesis scene
- calculate obs for all subVecEnv - call each config's obs computation function
- calculate reward for all subVecEnv - call each config's reward computation function
- reset all robots's to initial position
- apply action to all robots

How to prepare data required by joint training subVecEnv / shared network training subVecEnv
Joint training subVecEnv
- concat obs from different robot from each obs , which will keep the env dim length but will make obs dim longer
- for truncate / termination, use or logic, if one robot is terminated, the report terminated to all subVecEnv
Shared network training subVecEnv
- concat on env dim, which will make it looks like we have m robot *n envs =mn number of envs, but the obs dim will be the same as each robot 
- truncate / termination also longer dim (m*n)

IMPORTANT CHANGE 3

I wanna encode the way to compute observation and reward as a function into the robot config class, which takes the mother env class as input arg (this just meant to give it access to all information in the mother AEC class as well as the Genesis Scene). 


IMPORTANT CHANGE 4
Create special subVecEnv class for joint training and shared network training. 
For joint training, the agent will concat obs of different robot in each env so that the jointsubVecEnv will still get obs , rew, term, trucate... of n number of envs, but each obs is much longer. Also the same for action. Just, almost the same with normal basic subVecEnv
For shared network training, it have every other thing the same with normal subVecEnv, but the number of envs is not n_envs but m*n_envs, m is the number of robots in the bundle.


Let me clearly state the execution flow

1. AEC intialize stage
   1. Get all the input robot Configs call the subVecEnv configs
   3. Create queue to communicate with each subVecEnv
   2. create a function that spawn sub training environemnts in different threads. 
      1. For those basic robot configs, we create subVecEnvs   ; for joint training config, we create joint training subVecEnv; for shared network training config, we create shared network training subVecEnv;  
      2. The function does: create subVecEnv instances, reset it to get intial state cached, wrap the subVenEnv instance in onPolicyRunner of rsl-rl, and start traing by runner.learn. In this way we don't need the function to get immediate observation from mother env
      3. Those subVecEnv's output should be redirected to /log/run-name/subVecEnv-name.log, and if it does tensorboard, it should also follows this logic 
   2. create a list of all basic robot configs (extract all basic sub configs from those bundles)
   3. reorganize those basic robot configs into bundles with same frequency, create a list of bundle configs, call them agent configs
   4. Intialized state flags for each subVecEnv, all intially set to AEC side
   5. Create agent class from agent configs + corresponding state flags + reference to corresponding subVecEnv , ensure they are connected
   6. Setup things required by AEC API (include centuralized schedualer), map name to Agent class
   7. Create gs scene (including Calculate optimal Genesis frequency), add robots by calling _setup_robots function from each agent class
   8. Init buffers 

2. Initial Reset Stage
   1. AEC reset() 
   2. AEC  last(first_agent) - provide the observation required by subVecEnv to let them finish reset() -> obs + cache inital obs, set the flag to subVecEnv side
   3. AEC goes into step(first_agent), since the flag is set to subVecEnv side, it will be put into sleep to wait for the flag of first agent to set to AEC side again
   3. SubEnv-first_agent finish reset()
   4. SubEnv-first_agent start to intialize onPolicyRunner, call get_observation, which returns the cached obs (now the get_obs function should not consume the cache, keep it)
   5. SubEnv-first_agent finish initializing onPolicyRunner, calls get_observation again to get  inital obs, which also returns the cached obs (correct for intital obs)
   6. SubEnv-fist_agent goes into second step(), wait for the flag to set to SubEnv side
   6. OnpolicyRunner-first_agent infer action, call step in subVecEnv to send the action to AEC, those actions goes into action cache, and set the flag to AEC side
   7. AEC step(first_agent), since the flag is set to AEC side, it will be awaken and fetch the action from cache, apply it to genesis, but since the next agent is scheuled 0 frames later, no genesis scene step is performed
   8. AEC last(second_agent), ...
   .....
   9. Utill all subVecEnv finish their reset and goes into their second step() call


3. Training Stage
   1. AEC Env last() -> train() -> last() -> train() ...
   2. OnPolicyRunner in other threads -> step() -> step() -> step() -> step() ...


Big big problem, I think in current implementation, you mess up the idea of agent and subVecEnv and think they have one to one correspondance, but it is actually not! One agent may coorespond to serveral  subVecEnv! Let's change how we do _extract_and_reorganize_robot_configs. Don't extract, directly reorganize on the original robot configs by frequency, so that each agent knows which subVecEnv it is  correspond to. After recieve a list of original user input bundle, agent will store this list inside, as well as create a flatten version for easier inference in its init function.After it initialize, the  mother env can read the agent's falltened bundle and create indexed and etc ...

Requried Tests:
1. Read and understand how onPolicyRunner interact with subVecEnv, and create mocked onPolicyRunner class for testing (just follow the same order to call subVecEnv's API, but no training), so as to allow use to provide fixed action and check observation for testing
2. Create a AEC class with mocked _spawn_subvecenvs that uses the mocked onPolicyRunner


