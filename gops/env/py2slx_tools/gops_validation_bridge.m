function gops_validation_bridge(block)
setup(block);
%endfunction

function setup(block)
setup_python()

% Register number of ports
block.NumInputPorts  = 1;
block.NumOutputPorts = 1;

% Setup port properties to be inherited or dynamic
% block.SetPreCompInpPortInfoToDynamic;
% block.SetPreCompOutPortInfoToDynamic;
block.SetPreCompPortInfoToDefaults;

% Override input port properties
block.InputPort(1).Dimensions = -1;
block.InputPort(1).DatatypeID = 0;  % double
block.InputPort(1).Complexity = 'Real';
block.InputPort(1).DirectFeedthrough = true;

% Override output port properties
block.OutputPort(1).Dimensions = -1;
block.OutputPort(1).DatatypeID = 0; % double
block.OutputPort(1).Complexity = 'Real';

% Register parameters
block.NumDialogPrms = 1;
block.DialogPrmsTunable = {'Nontunable'};

block.SampleTimes = [-1 0];  % Inherited by default, [Ts 0] for discrete

block.SimStateCompliance = 'DefaultSimState';

block.RegBlockMethod('SetInputPortDimensions', @SetInputPortDimensions);
block.RegBlockMethod('CheckParameters', @CheckParameters);
block.RegBlockMethod('Outputs', @Outputs); % Required
block.RegBlockMethod('Terminate', @Terminate); % Required

function SetInputPortDimensions(block, ~, inputDim)
if isscalar(inputDim)
    dummy_obs = zeros(1, inputDim);
else
    dummy_obs = zeros(inputDim);
end
act = inference(block, dummy_obs);
if isscalar(act) || isvector(act)
    outputDim = length(act);
else
    outputDim = size(act);
end
block.InputPort(1).Dimensions = inputDim;
block.OutputPort(1).Dimensions = outputDim;
%end SetInputPortDimensions

function CheckParameters(block)
model_path = block.DialogPrm(1).Data;
load_model(block, model_path)
%end SetOutputPortDimensions

function Outputs(block)
obs = block.InputPort(1).Data;
act = inference(block, obs);
block.OutputPort(1).Data = act;
%end Outputs

function Terminate(block)
set_param(block.BlockHandle, 'UserData', []);
%end Terminate

% Utility functions

function setup_python()
setup_script = {
    'import functools, numpy as np, torch, torch.jit'
    'def model_inference(model, obs):'
    '  obs = torch.from_numpy(obs)'
    '  with torch.no_grad():'
    '    act = model(obs)'
    '  return act.numpy()'
    'def load_model_for_inference(path):'
    '  model = torch.jit.load(model_path)'
    '  return functools.partial(model_inference, model)'
};
pyrun(setup_script)
%end setup_python

function load_model(block, model_path)
load_script = 'inference_fn = load_model_for_inference(model_path)';
inference_fn = pyrun(load_script, 'inference_fn', 'model_path', model_path);
set_param(block.BlockHandle, 'UserData', inference_fn);
%end load_model

function act = inference(block, obs)
inference_fn = get_param(block.BlockHandle, 'UserData');
act = inference_fn(py.numpy.array(single(obs)));
act = cast(act, 'like', obs);
%end inference
