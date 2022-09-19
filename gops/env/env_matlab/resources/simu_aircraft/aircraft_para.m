%  Copyright (c) Intelligent Driving Lab(iDLab), Tsinghua University. All Rights Reserved.
%
%  Creator: Baiyu Peng
%  Description: load parameters for cartpole simulink model (cartpole.slx)
%  Update Date: 2021-6-8, Baiyu Peng: first version

%  General Optimal control Problem Solver (GOPS)

clc;clear;close all;
set_param('aircraft', 'PostCodeGenCommand', 'mat2json');
% model parameters
Beta = 426.4352;
Gamma = 0.01;
Ka = 0.677;
Kf = -1.746;
Ki = -3.864;
Kq = 0.8156;
Md = -6.8847;
Mq = -0.6571;
Mw = -0.00592;
Sa = 0.005236;
Swg = 3;
Ta = 0.05;
Tal = 0.3959;
Ts = 0.1;
Uo = 689.4;
Vto = 690.4;
W1 = 2.971;
W2 = 4.144;
Wa = 10;
Zd = -63.9979;
Zw = -0.6385;
a = 2.5348;
b = 64.13;
cmdgain = 0.034909544727280771;
g = 32.2;

% action space range,don't use 'inf'
a_min = [-1.,];
a_max = [1.,];

% adver space range,don't use 'inf'
adva_min = [-0.1,];
adva_max = [0.1,];

% state space range,don't use 'inf'
x_min = [-9999, -1.5,];
x_max = [9999, 1.5,];

% initial state 
% (Note:the initial state will be re-assigned in python, so you can designate any value here. This step is only used to generate C++ code)
w_o = 0;
q_o = 0;

x_ini = [w_o, q_o];
