import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt
import tensorflow as tf
import tf_agents
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import wrappers
from tf_agents.policies import random_tf_policy
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers.schedules import ExponentialDecay, PolynomialDecay
from environment import ship_environment
import os
import time

# collecting data with random policy to populate replay buffer
def collect_init_data(environment, policy, buffer):
    time_step = environment.reset()
    episode_return = 0
    while not np.equal(time_step.step_type, 2):

        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        episode_return += next_time_step.reward.numpy()[0]
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        buffer.add_batch(traj)
        time_step=next_time_step

    return episode_return

# collect data through epsilon-greedy policy and train agent
def collect_data(envr, policy, buffer, dataset, agent, ep_counter, params):
    time_step = envr.reset()
    episode_return = 0
    global timestep_counter

    # time_step.step_type:  0-> initial step,1->intermediate, 2-> terminal step
    while not np.equal(time_step.step_type, 2):

        action_step = policy.action(time_step)

        #using custom annealing epsilon-greedy policy to collect training data
        epsilon = params.epsilon_greedy(ep_counter)
        if np.random.random() < epsilon:
            action_no = np.random.randint(0, 3)
            action = tf.constant([action_no],  shape=(1,), dtype=np.int64,name='action')
            action_step = tf_agents.trajectories.policy_step.PolicyStep(action)

        next_time_step = envr.step(action_step)

        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        episode_return = next_time_step.reward.numpy()[0] + episode_return
        buffer.add_batch(traj)
        time_step = next_time_step

        #update q-network once every some steps
        if timestep_counter % params.DQN_update_time_steps == 0:
            iterator=iter(dataset)
            experience, unused_info = next(iterator)
            agent.train(experience)

        timestep_counter+=1
    print("EPISODE RETURN", episode_return)
    return episode_return, agent

def dqn_train(params):
    mname = params.model_name
    if not os.path.exists('saved_models/' + mname):
        if not os.path.exists('saved_models'):
            os.mkdir('saved_models')
        os.mkdir('saved_models/' + mname)
        os.mkdir('saved_models/' + mname + '/plots')

    params.print_params('saved_models/' + mname)

    env = wrappers.TimeLimit(ship_environment(train_test_flag=0, wind_flag=params.wind_flag,
                                              wind_speed=params.wind_speed, wind_dir=params.wind_dir,
                                              wave_flag=params.wave_flag, wave_height=params.wave_height,
                                              wave_period=params.wave_period, wave_dir=params.wave_dir),
                             duration=params.duration)
    tf_env = tf_py_environment.TFPyEnvironment(env)

    #learning rate scheduler to decay learning rate with time
    lr_schedule=ExponentialDecay(initial_learning_rate=params.initial_learning_rate,
                                 decay_steps=params.decay_steps,
                                 decay_rate=params.decay_rate)
    # lr_schedule = PolynomialDecay(initial_learning_rate=0.01, decay_steps=50000, end_learning_rate=0.001, power=5)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    q_net = q_network.QNetwork(tf_env.observation_spec(),tf_env.action_spec(),
                               fc_layer_params=params.fc_layer_params,
                               activation_fn=tf.keras.activations.tanh)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        gamma=params.discount_factor,
        target_update_tau=params.target_update_tau,
        target_update_period=params.target_update_period,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=params.replay_buffer_max_length)

    random_policy = random_tf_policy.RandomTFPolicy(tf_env.time_step_spec(),tf_env.action_spec())

    #collecting samples with random policy
    for __ in range(10):
        collect_init_data(tf_env, random_policy, replay_buffer)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=params.num_parallel_calls,
        sample_batch_size=params.sample_batch_size,
        num_steps=params.num_steps).prefetch(params.prefetch)

    # Reset the train step
    agent.train_step_counter.assign(0)

    ep_counter=0
    global timestep_counter
    timestep_counter=0

    episodes = params.max_episodes
    returns = []
    train_loss = []

    while ep_counter < episodes:

        ereturn, agent = collect_data(tf_env, agent.policy, replay_buffer, dataset, agent, ep_counter, params)
        returns.append(ereturn)

        # Sample a batch of data from the buffer and update the agent's network.
        iterator = iter(dataset)
        experience, unused_info = next(iterator)
        eloss = agent.train(experience).loss
        train_loss.append(eloss)

        if len(train_loss) < 110:
            print('episode = {0}, Loss = {1}'.format(ep_counter, eloss))
        else:
            print('episode = {0}, Avg Loss (100 ep) = {1}'.format(ep_counter,
                                                                  (sum(train_loss[-100:-1]) + train_loss[-1]) / 100))

        ep_counter +=1

        # saving agent's policy at intervals
        if ep_counter % params.DQN_policy_store_frequency == 0 and ep_counter >= 1:
            policy_dir = os.path.join('saved_models', mname, mname + '_ep_' + str(ep_counter))
            tf_policy_saver = policy_saver.PolicySaver(agent.policy)
            tf_policy_saver.save(policy_dir)

    # checkpoint_dir = 'saved_models/jan8_'
    # train_checkpointer = common.Checkpointer(
    #     ckpt_dir=checkpoint_dir,
    #     max_to_keep=1,
    #     agent=agent,
    #     policy=agent.policy,
    #     replay_buffer=replay_buffer,
    # )

    train_loss = np.array(train_loss)
    interval = params.DQN_loss_avg_interval
    avg_losses, avg_returns = [], []

    for i in range(len(train_loss) - interval):
        avg_returns.append(sum(returns[i:i + interval]) / interval)
        avg_losses.append(sum(train_loss[i:i + interval]) / interval)

    mat_dict = {'returns':np.array(returns), 'loss':np.array(train_loss),
                'avg_returns':np.array(avg_returns), 'avg_loss':np.array(avg_losses)}

    savemat('saved_models/'+mname+'/'+mname+'.mat', mat_dict)

    plt.figure()
    plt.title("Returns vs. Episodes")
    plt.ylabel("Returns")
    plt.plot(avg_returns)
    plt.xlabel("Episodes")
    plt.grid()
    plt.savefig('saved_models/'+mname+'/plots/'+mname+'_return.png', dpi=600)

    plt.figure()
    plt.title("Train-loss vs. Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.plot(avg_losses)
    plt.grid()
    plt.savefig('saved_models/'+mname+'/plots/'+mname+'_loss.png', dpi=600)
    # plt.show()

if __name__ == "__main__":
    import hyperparams as params
    dqn_train(params)
