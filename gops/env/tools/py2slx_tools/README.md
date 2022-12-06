# GOPS validation bridge
Put GOPS policy back into Simulink for closed-loop validation, with minimum effort.

## Prerequisites
1. MATLAB with Simulink.

    We recommend the up-to-date MATLAB version. Since the bridge uses `pyrun`, the minimum MATLAB version is R2021b.

2. Python installation compatible with MATLAB. 

   Check MATLAB documentation [Configure Your System to Use Python](https://www.mathworks.com/help/releases/R2022a/matlab/matlab_external/install-supported-python-implementation.html), especially "Versions of Python Compatible with MATLAB Products by Release" in "Related Topics" section.

## Usage
1. Users can refer to the example of document `py2slx_example.py` to use the policy conversion tool.

    You need to set four parameters(`log_policy_dir_list`、`trained_policy_iteration_list`、`export_controller_name`、`save_path`) according to the requirements of the reference example,.


2. Run the example file you configured above.

   Py2slx tool will check the compatibility of the model to confirm whether it can be converted and whether user's matlab version meets the requirements.
   
   - GOPS builtin models should work well without compatibility issue. If meeting something difficult, you could refer to [PyTorch JIT documentation](https://pytorch.org/docs/stable/jit.html) for requirements about `trace-able nn.Module`.
   - If matlab is not installed on your computer or the version is incorrect, the corresponding prompt will appear.
   - If everything is OK, a saved model will exist at `save_path` and the latest version of matlab on your computer will be opened.

---

3. (The following steps happen in MATLAB) Launch MATLAB (If the matlab opened in the previous step is not turned off by you, you can ignore this step.) 
    
   You should make sure launching MATLAB in a Python environment with `PyTorch` installed. The Python environment where `GOPS` installed in should be OK, but if you have any difficulty (e.g. on another machine), `PyTorch` is the only requirement.

    - If you prefer your system-wide Python installation, it's ok to launch MATLAB either from shortcut or commandline.

    - If you use a conda-based environment, the most convenient way is to launch MATLAB in commandline

        ```shell
        conda activate <YOUR_ENV_WITH_PYTORCH>
        matlab
        ```
    
    - If you use other type of environments, the activation method may vary, but the most important thing is to keep environment variables correct (same as when you launch a Python script in that environment) when launching MATLAB.

    To verify the environment is correct, you could type `pyenv` in MATLAB Command Window. The `Version`, `Executable`, `Library` and `Home` field of the result should match your target Python environment.

4. Copy `gops_validation_bridge.m` to your Simulink model directory. Create a `Level-2 MATLAB S-Function` block, set:
    - `S-Function Name` field to `gops_validation_bridge`
    - `Parameters` field to `'save_path'` (e.g. `'model.pt'`). The path is relative to your MATLAB working directory and an absolute path will also work.
    
    The observation will be fed into the block inport, and the block outport will output action from trained policy.

5. Connect the `Level-2 MATLAB S-Function` block as the closed-loop controller in your Simulink model. And start your simulation and validation.
