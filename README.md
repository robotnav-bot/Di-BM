[[Paper]](https://arxiv.org/abs/2601.12397) [[Project Page]](https://robotnav-bot.github.io/Di-BM-project/)
# Learning Diverse Skills for Behavior Models with Mixture of Experts


This repository contains the implementation of the Di-BM policy within the [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) framework. Follow the steps below to set up the environment, collect data, train the model, and evaluate the policy.

## 1. Installation & Setup

### 1.1 Download Repository
Download [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) and clone our repository into the `RoboTwin/policy/` directory.

### 1.2 Environment Configuration
1. **Base Environment:** Refer to [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) instructions to configure the base Conda/Python environment.
2. **Install Dependencies for Di-BM:** Navigate to the policy directory and install the required dependencies:

```bash
cd policy/Di-BM

# Install specific versions of dependencies
pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor sympy

# Install the current package in editable mode
pip install -e .
```

---

## 2. Data Collection & Processing

### 2.1 Collect Single-Task Datasets
**Note:** All data collection scripts should be executed from the root `RoboTwin/` directory.

Use the `collect_data.sh` script to generate data for a specific task.


```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
```

**Example:**
```bash
bash collect_data.sh beat_block_hammer demo_clean 0
```

### 2.2 Synthesize Multi-Task Zarr Dataset
Once single-task datasets are collected, process them into a multi-task Zarr format.


```bash
cd policy/Di-BM
bash process_data.sh ${task_names} ${task_config} ${expert_data_num}
```

**Example:**
```bash
bash process_data.sh adjust_bottle click_bell demo_clean 50
```
The Zarr dataset is located at `/data/multi_task-demo_clean-50.zarr`.

---

## 3. Train Policy

To train the policy, use the `train.sh` script in `policy/Di-BM`.


```bash
bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${action_dim} ${gpu_id}
```

**Example (Multi-task training):**
The following example uses the configuration from `./diffusion_policy/config/robot_dp_14.yaml`:

```bash
bash train.sh multi_task demo_clean 50 0 14 0
```

---

## 4. Evaluate Policy

### 4.1 Compute Z
First, calculate the `Z` values for the experts using the `compute_Z.sh` script.


```bash
bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id} ${ckpt_path}
```

**Example:**
```bash
bash compute_Z.sh multi_task demo_clean 50 0 14 0 ./checkpoints/multi_task-demo_clean-50-0/500.ckpt
```

### 4.2 Update Policy Code
**⚠️ Manual Step Required:**
1. Copy the tensor output from Step 4.1 (e.g., `[[18.739338, 23.333763, ...]]`).
2. Open the file `./diffusion_policy/policy/diffusion_unet_image_policy.py`.
3. Locate **line 196**.
4. Paste the values to the `self.Z` variable:

```python
# ./diffusion_policy/policy/diffusion_unet_image_policy.py

# Replace the Z tensor with your computed values:
self.Z = torch.Tensor([[18.739338, 23.333763, 13.635132, 6.825393, 76.133156]]).to(global_cond.device)
```

### 4.3 Configure & Run Evaluation
1. **Set Checkpoint:** Open `./deploy_policy.yml` and set the `checkpoint_num` variable to the specific checkpoint you wish to evaluate.
2. **Run Evaluation:**


```bash
bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id}
```

**Example:**
```bash
bash eval.sh adjust_bottle demo_clean demo_clean 50 0 0
```

The results will be shown in `RoboTwin/eval_result`


## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{shen2026learningdiverseskillsbehavior,
      title={Learning Diverse Skills for Behavior Models with Mixture of Experts}, 
      author={Wangtian Shen and Jinming Ma and Mingliang Zhou and Ziyang Meng},
      year={2026},
      eprint={2601.12397},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2601.12397}, 
}