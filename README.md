# Deep-Reinforcement-Learning-Based-Control-for-Ship-Navigation

Achieves path following of an autonomous surface vessel (ASV) with a Deep Q-Network (DQN) agent. This code uses reinforcement learning to train an agent to control the rudder angle of the Kriso Container Ship (KCS) to achieve waypoint tracking the presence of calm waters and in the presence of winds

# Why the project is useful?
A majority of marine accidents that occur can be attributed to errors in human decisions. Through automation, the occurrence of such incidents can be minimized. Therefore, automation in the marine industry has been receiving increased attention in the recent years. This work investigates the automation of the path following action of a ship. A deep Q-learning approach is proposed to solve the path-following problem of a ship. This method comes under the broader area of deep reinforcement learning (DRL) and is well suited for such tasks, as it can learn to take optimal decisions through sufficient experience. This algorithm also balances the exploration and the exploitation schemes of an agent operating in an environment. A three-degree-of-freedom (3-DOF) dynamic model is adopted to describe the ship’s motion. The Krisco container ship (KCS) is chosen for this study as it is a benchmark hull that is used in several studies and its hydrodynamic coefficients are readily available for numerical modeling. Numerical simulations for the turning circle and zig-zag maneuver tests are performed to verify the accuracy of the proposed dynamic model. A reinforcement learning (RL) agent is trained to interact with this numerical model to achieve waypoint tracking. Finally, wind forces are modelled and the performance of the RL based controller is evaluated in the presence of wind.

# What does the project do?
This repository contains code for implementing reinforcement learning based control for the path following of autonomous ships. The code is trained in a docker container to maintain easy portability.

# How to get started with the project?
The project has been setup with a docker file that can be used to build the docker container in which the code will be executed. It is presumed that you have docker installed on your system. Please follow the following steps to get the code up and running.

**Prerequisites:**

1. An Ubuntu OS (has been tested on Ubuntu 20.04). Note this will not work on Windows OS systems as x11 forwarding needs to be handled separately on it and also NVIDIA GPU support for docker containers as of now is not available on Windows.
2. Docker installed on your system ([Link](https://docs.docker.com/engine/install/ubuntu/))
3. Complete the steps to use docker as a non-root user ([Link](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user))
4. NVIDIA driver is installed (for GPU use) - typing `nvidia-smi` on a terminal should tell you if you have the NVIDIA drivers or not (If not follow instructions at this [Link](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html))
5. NVIDIA Container Toolkit is installed to interface GPUs to docker containers ([Link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit))

**Note:** Some of these links listed above may move over time. In such a case please find the appropriate links to find the instructions for installation.

**Step 1:** 

Clone the repository to your system

```commandline 
git clone https://github.com/MarineAutonomy/Deep-Reinforcement-Learning-Based-Control-for-Ship-Navigation.git
```

and cd into the src folder. The prompt on the terminal should look as ```.../DQN-ASV-path-follow/src$```

**Step 2:** 

Change shell scripts to be executables 
```commandline
chmod +x docker.sh
chmod +x run_docker.sh
```

Execute the script docker.sh to build the docker container

```commandline
./docker.sh 
```

**Step 3:** 

Type the following command 

```commandline
xauth list
```
and you will see a response similar (you may see more lines on your computer) to 

```
iit-m/unix:  MIT-MAGIC-COOKIE-1  958745625f2yu22358u3ebe5cc4ad453
#ffff#6969742d6d#:  MIT-MAGIC-COOKIE-1  958745625f2yu22358u3ebe5cc4ad453
```

Copy the first line of the response that corresponds to your computer name (as seen after @ in the prompt on a terminal). So for ```user@iit-m$``` displayed on the prompt the computer name is ```iit-m```.

**Step 4:** 

Run the docker container

```commandline
./run_docker.sh 
```

Notice that this binds the current /src directory on the host machine to the /src directory on the container. Thus, the code can be edited on the host machine and the changes will be reflected in the container instantly. 

**Step 5:** 

Type the following command inside the container (Check that the terminal prompt reflects something like ```docker@iit-m:~/DQN-ASV-path-follow/src$``` with ```iit-m``` replaced by your computer name)

```commandline
xauth add <your MIT-MAGIC-COOKIE-1>
```
where make sure to replace ```<your MIT-MAGIC-COOKIE-1>``` with your cookie that you copied in step 3. Notice however, that a ```0``` must be added at the end of ```<computer-name/unix:>```

```
xauth add iit-m/unix:0  MIT-MAGIC-COOKIE-1  958745625f2yu22358u3ebe5cc4ad453
```

**Step 6:** 
Now the files should be executable inside the docker container

# Important python scripts

**kcs folder** 

```environment.py``` has the python environment description for the path following of KCS. Edit this file to modify the dynamics of the vehicle and how it interacts with the environment. Currently this study uses the 3-DOF MMG model to mimic the dynamics of the KCS vessel in calm waters and in wind. To start a single training run, edit the training hyperparameters in ```hyperparams.py``` execute the file ```dqn_train.py``` inside the docker container. Before starting a training run, ensure that the the model and any plots are getting saved in the correct directories. To test various trajectories including single waypoint tracking, elliptical trajectory and other trajectories in calm water or in wind run ```dqn_test.py``` (make sure to load the correct model). If you want to sequentially train multiple training runs, then set the required hyperparameters in ```batch_train.py``` and execute ```batch_train.py``` inside the docker container.  

# Where can you get help with your project?
Help on this project can be sought by emailing R S Sanjeev Kumar at `sanjeevrs2000@gmail.com` or Md Shadab Alam at `shaadalam.5u@gmail.com` or the supervisor Abhilash Somayajula at `abhilash@iitm.ac.in`. For any proposed updates to the codes please raise a pull request.

# Who maintains and contributes to the project?
This project is currently maintained by Marine Autonomous Vessels (MAV) Laboratory within the department of Ocean Engineering at IIT Madras. The project has originally been undertaken by Rohit Deraj, R S Sanjeev Kumar, Md Shadab Alam and has been supervised by Abhilash Somayajula. Contributions to the repository have been made by R S Sanjeev Kumar, Md Shadab Alam and Abhilash Somayajula.
