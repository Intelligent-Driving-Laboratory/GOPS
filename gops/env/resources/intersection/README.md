# RL environment for large-scale autonomous driving tasks
<div  align="center">    
<img src="utils/illustration.jpg" alt="drawing" height="420" width="600"/>
</div>

This repository aims to build a general and extensible RL training and testing environment for 
the autonomous driving tasks. It is highlighted in the following way:

1) We incorporate a sophisticated 
vehicle dynamics and the large-scale traffic and map constructed using
SUMO to realize co-simulation of the ego vehicle and its surrounding 
environment. 

2) We design the basic RL elements in the field of
autonomous driving following the interface defined in Gym, such as the state,
the action, the reward function, and the reset conditions etc, which can
be easily reused or modified depending on your task (see [here](https://github.com/mahaitongdae/safe_exp_env) for an example).

3) Correspondingly, we develop an analytic model of the environment to facilitate its use
in model-based RL algorithms. 

4) We also include a bunch of training and testing environments that we use, for instance, the one for
[integrated decision and control](https://arxiv.org/pdf/2103.10290.pdf) (see branch ```master```); the one for
[centralized coordination](https://arxiv.org/pdf/1912.08410.pdf) (see branch ```centralized_env```); ones
with specific designed RL elements such as using bird view image as state (```cross_nocontrol_fixedgrid_end2end```),
or using trajectory as action (```toyota202003```); and ones with different vehicle parameters, or the map
(e.g. ```toyota202012```, ```toyota202012_3lane```, ```toyota202103_exp```).

## Getting Started
### Prerequisites
* [SUMO](https://sumo.dlr.de/docs/Downloads.php) >= 1.7.0

* Python >= 3.6

* [Gym](https://sumo.dlr.de/docs/Downloads.php) >= 1.17.3

* [Tensorflow](https://www.tensorflow.org/install) >= 2.2

### Install
1. Download this repo on your host.

2. Register this environment (see [here](https://github.com/openai/gym/issues/626) or follow the below steps).
    * Create a python package, say ```user_defined```, under your installed gym path ```/path/to/gym/envs/```
    
    * Copying the downloaded files ```endtoend.py```, 
    ```dynamics_and_models.py```, ```endtoend_env_utils.py```, 
    ```traffic.py```, together with the whole directory ```sumo_files```
    to ```/path/to/gym/envs/user_defined```.
  
    * Add a line in ```/path/to/gym/envs/user_defined/__init__.py```:
    ```from gym.envs.user_defined.endtoend import CrossroadEnd2end```
    
    * Add the following code in the end of the file ```/path/to/gym/envs/__init__.py```
        ```
        register(
            id='CrossroadEnd2end-v0',
            entry_point='gym.envs.user_defined:CrossroadEnd2end',
            max_episode_steps=200,
        )
        ```
### How to use
Use it just as the other embedded gym environments
```
env = gym.make('CrossroadEnd2end-v0')
```

## Reference

This environment is officially proposed in the paper
[integrated decision and control](https://arxiv.org/pdf/2103.10290.pdf)
and here's a BibTeX entry that you can use to cite it in a publication.
```
@article{guan2021integrated,
  title={Integrated Decision and Control: Towards Interpretable and Efficient Driving Intelligence},
  author={Guan, Yang and Ren, Yangang and Li, Shengbo Eben and Ma, Haitong and Duan, Jingliang and Cheng, Bo},
  journal={arXiv preprint arXiv:2103.10290},
  year={2021}
}
```





