%  Copyright (c) Intelligent Driving Lab(iDLab), Tsinghua University. All Rights Reserved.
%
%  Creator: Baiyu Peng
%  Description: load parameters for cartpole simulink model (cartpole.slx)
%  Update Date: 2021-6-8, Baiyu Peng: first version

%  General Optimal control Problem Solver (GOPS)

clc;clear;close all;
set_param('doublemass', 'PostCodeGenCommand', 'mat2json');
% model parameters
ka = 50;
m1 = 10;
m2 = 10;


% action space range,don't use 'inf'
a_min = [-10,];
a_max = [10,];

% adver space range,don't use 'inf'
adva_min = [-0.001,-0.001,-0.001,-0.001,];
adva_max = [0.001,0.001,0.001,0.001,];

% state space range,don't use 'inf'
x_min = [-9999, -9999, -9999, -9999,];
x_max = [9999, 9999, 9999, 9999,];

% initial state 
% (Note:the initial state will be re-assigned in python, so you can designate any value here. This step is only used to generate C++ code)
x1_o = -0.1;
x1dot_o = 0;
x2_o = 0.1;
x2dot_o = 0;

x_ini = [x1_o, x1dot_o, x2_o, x2dot_o,];

