# GOPS validation bridge
Put GOPS policy back into Simulink for closed-loop validation, with minimum effort.

## Prerequisites
1. MATLAB with Simulink.

    We recommend the up-to-date MATLAB version. Since the bridge uses `pyrun`, the minimum MATLAB version is R2021b.

2. Python installation compatible with MATLAB. 

    Check MATLAB documentation [Configure Your System to Use Python](https://www.mathworks.com/help/releases/R2022a/matlab/matlab_external/install-supported-python-implementation.html), especially "Versions of Python Compatible with MATLAB Products by Release" in "Related Topics" section.

## Usage
1. In your training file, put the following snippet after a model is created.

    ```python
    from gops.env.py2slx.export import check_jit_compatibility
    check_jit_compatibility(model, example_obs)
    ```
    where `model` is the apprfunc you created and `example_obs` is an **unbatched** `torch.Tensor` consistent with environment observation space (in other words, `example_obs.shape` should be same as `observation_space.shape`). The content of `example_obs` is abitrary (all zero or random or real obs are both OK), as long as your model do not complain during inference.

    Running this piece of code catches any incompatibility early, avoiding wasting time after training process. GOPS builtin models should work well without compatibility issue. If meeting something difficult, you could refer to [PyTorch JIT documentation](https://pytorch.org/docs/stable/jit.html) for requirements about `trace-able nn.Module`.

2. Put the following snippet after the traing loop is finished (or anywhere you would like to checkpoint).

    ```python
    from gops.env.py2slx import export_model
    export_model(model, example_obs, save_path)
    ```
    where the meaning of `model` and `example_obs` are same as the previous step, and `save_path` is a path like `model.pt` where the model will be saved at.

3. Train your algorithm as usual. If everything is OK, a saved model will exist at `save_path`.

---

4. (The following steps happen in MATLAB) Launch MATLAB. You should make sure launching MATLAB in a Python environment with `PyTorch` installed. The Python environment where `GOPS` installed in should be OK, but if you have any difficulty (e.g. on another machine), `PyTorch` is the only requirement.

    - If you prefer your system-wide Python installation, it's ok to launch MATLAB either from shortcut or commandline.

    - If you use a conda-based environment, the most convenient way is to launch MATLAB in commandline

        ```shell
        conda activate <YOUR_ENV_WITH_PYTORCH>
        matlab
        ```
    
    - If you use other type of environments, the activation method may vary, but the most important thing is to keep environment variables correct (same as when you launch a Python script in that environment) when launching MATLAB.

    To verify the environment is correct, you could type `pyenv` in MATLAB Command Window. The `Version`, `Executable`, `Library` and `Home` field of the result should match your target Python environment.

5. Copy `gops_validation_bridge.m` to your Simulink model directory. Create a `Level-2 MATLAB S-Function` block, set:
    - `S-Function Name` field to `gops_validation_bridge`
    - `Parameters` field to `'save_path'` (e.g. `'model.pt'`). The path is relative to your MATLAB working directory and an absolute path will also work.
    
    The observation will be fed into the block inport, and the block outport will output action from trained policy.

6. Connect the `Level-2 MATLAB S-Function` block as the closed-loop controller in your Simulink model. And start your simulation and validation.
