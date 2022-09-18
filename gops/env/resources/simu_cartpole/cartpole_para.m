%  Copyright (c) Intelligent Driving Lab(iDLab), Tsinghua University. All Rights Reserved.
%
%  Creator: Baiyu Peng
%  Description: load parameters for cartpole simulink model (cartpole.slx)
%  Update Date: 2021-6-8, Baiyu Peng: first version

%  General Optimal control Problem Solver (GOPS)

clc;clear;close all;
set_param('cartpole', 'PostCodeGenCommand', 'mat2json');
% model parameters
m_cart = 1;
m_pendulum = 0.1;
l_cg = 0.5;
g = 9.8;

% action space range,don't use 'inf'
a_min = [-30];
a_max = [30];

% adver space range,don't use 'inf'
adva_min = [-1];
adva_max = [1];

% state space range,don't use 'inf'
x_min = [-4.8, -9999, -0.4, -9999];
x_max = [4.8, 9999, 0.4, 9999];

% initial state 
% (Note:the initial state will be re-assigned in python, so you can designate any value here. This step is only used to generate C++ code)
x_o = 0;
xdot_o = 0;
theta_o = 0;
thetadot_o = 0;

x_ini = [x_o, xdot_o, theta_o, thetadot_o];

