"""Utility script to extract Unitree Quadruped policy network to ONNX without Genesis dependencies"""

import argparse
import json
import os
import pickle
from collections import OrderedDict

import onnx
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-Layer Perceptron matching RSL-RL's ActorCritic.actor structure"""

    def __init__(self, input_dim, hidden_dims, output_dim, activation='elu'):
        super().__init__()

        # Build the network layers
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # Add activation except for the last layer
            if i < len(dims) - 2:
                if activation == 'elu':
                    layers.append(nn.ELU())
                elif activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                else:
                    raise ValueError(f"Unknown activation: {activation}")

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def extract_actor_weights(checkpoint_path):
    """Extract just the actor network weights from checkpoint"""
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract actor weights
    actor_state_dict = OrderedDict()
    for key, value in checkpoint['model_state_dict'].items():
        if key.startswith('actor.'):
            # Remove 'actor.' prefix and rename to match Sequential module
            new_key = key.replace('actor.', '')
            actor_state_dict[new_key] = value

    # Also get the action std if available
    if 'std' in checkpoint['model_state_dict']:
        std = checkpoint['model_state_dict']['std']
        print(f"  - Action std shape: {std.shape}")
    else:
        std = None

    print(f"  - Found {len(actor_state_dict)} actor layers")
    return actor_state_dict, std, checkpoint.get('iter', 0)


def create_policy_network(obs_dim, action_dim, hidden_dims, activation='elu'):
    """Create the policy network architecture"""
    print(f"\n🏗️ Creating policy network:")
    print(f"  - Input dimension: {obs_dim}")
    print(f"  - Hidden dimensions: {hidden_dims}")
    print(f"  - Output dimension: {action_dim}")
    print(f"  - Activation: {activation}")

    policy = MLP(obs_dim, hidden_dims, action_dim, activation)
    return policy


def export_to_onnx(model, obs_dim, output_path, robot_type, std=None, control_freq=None):
    """Export the policy network to ONNX format with comprehensive documentation"""
    print(f"\n📦 Exporting to ONNX: {output_path}")

    # Calculate dt from control_freq if available
    dt = 1.0 / control_freq if control_freq else None

    # Create comprehensive documentation string for the ONNX model
    doc_string = f"""{robot_type.upper()} Quadruped Locomotion Policy Network

OBSERVATION SPACE (45 dimensions):
┌─────────────────────────────────────────────────────────────────────────────┐
│ Index │ Component          │ Dim │ Description                    │ Units   │
├─────────────────────────────────────────────────────────────────────────────┤
│ [0:3] │ Angular Velocity   │  3  │ Base angular velocity in       │ rad/s   │
│       │                    │     │ robot frame (roll, pitch, yaw) │         │
├─────────────────────────────────────────────────────────────────────────────┤
│ [3:6] │ Projected Gravity  │  3  │ Gravity vector projected into  │ unit vec│
│       │                    │     │ robot frame (for orientation)  │         │
├─────────────────────────────────────────────────────────────────────────────┤
│ [6:9] │ Command Velocities │  3  │ Desired velocities:            │ m/s,    │
│       │                    │     │ [vx, vy, yaw_rate]             │ rad/s   │
├─────────────────────────────────────────────────────────────────────────────┤
│ [9:21]│ Joint Positions    │ 12  │ Current joint angles relative  │ rad     │
│       │                    │     │ to default pose                │         │
├─────────────────────────────────────────────────────────────────────────────┤
│[21:33]│ Joint Velocities   │ 12  │ Current joint angular          │ rad/s   │
│       │                    │     │ velocities                     │         │
├─────────────────────────────────────────────────────────────────────────────┤
│[33:45]│ Previous Actions   │ 12  │ Previous motor commands        │ rad     │
│       │                    │     │ (for temporal consistency)     │         │
└─────────────────────────────────────────────────────────────────────────────┘

Joint Order (for positions, velocities, actions):
[FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
 RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]

ACTION SPACE (12 dimensions):
- Motor position commands (rad) relative to default pose
- Scaled by action_scale (0.25) in environment
- Applied as: target_pos = action * 0.25 + default_pose

SCALING FACTORS (applied in observation):
- Angular velocity: * 0.25
- Linear velocity commands: * 2.0
- Joint positions: * 1.0 (relative to default)
- Joint velocities: * 0.05

DEFAULT JOINT ANGLES (rad):
- Hip joints (FR, FL, RR, RL): 0.0
- Thigh joints (FR, FL): 0.8, (RR, RL): 1.0
- Calf joints (all): -1.5

CONTROL PARAMETERS:
- PD gains: kp=20.0, kd=0.5
- Control frequency: {f'{control_freq}Hz (dt={dt:.3f}s)' if control_freq else 'Configurable'}
- Action latency: 1 step (simulated)"""

    # Set model to eval mode
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, obs_dim)

    # Test forward pass
    with torch.no_grad():
        test_output = model(dummy_input)
        print(f"  - Test output shape: {test_output.shape}")

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['action_mean'],
        dynamic_axes={'observation': {0: 'batch_size'}, 'action_mean': {0: 'batch_size'}},
        verbose=False,
    )

    print(f"✅ Exported ONNX model to: {output_path}")

    # Try to validate and add metadata if onnx is available
    try:
        # Load the model
        onnx_model = onnx.load(output_path)

        # Add the comprehensive documentation string
        onnx_model.doc_string = doc_string

        # Add custom metadata properties
        metadata = [
            ("model_type", "quadruped_locomotion_policy"),
            ("robot", robot_type.upper()),
            ("framework", "Genesis"),
            ("observation_dim", str(obs_dim)),
            ("action_dim", "12"),
            (
                "joint_order",
                "FR_hip,FR_thigh,FR_calf,FL_hip,FL_thigh,FL_calf,RR_hip,RR_thigh,RR_calf,RL_hip,RL_thigh,RL_calf",
            ),
            (
                "observation_components",
                "ang_vel[0:3],proj_gravity[3:6],commands[6:9],joint_pos[9:21],joint_vel[21:33],prev_actions[33:45]",
            ),
            ("scaling_ang_vel", "0.25"),
            ("scaling_lin_vel", "2.0"),
            ("scaling_joint_pos", "1.0"),
            ("scaling_joint_vel", "0.05"),
            ("action_scale", "0.25"),
            ("control_frequency", f"{control_freq}Hz" if control_freq else "Configurable"),
            ("pd_gains", "kp=20.0,kd=0.5"),
            ("default_pose", "hip=0.0,thigh_front=0.8,thigh_rear=1.0,calf=-1.5"),
        ]

        for key, value in metadata:
            onnx_model.metadata_props.append(onnx.StringStringEntryProto(key=key, value=value))

        # Add descriptions to input/output tensors
        onnx_model.graph.input[0].doc_string = "Robot observation vector (45D): sensor readings and state information"
        onnx_model.graph.output[0].doc_string = (
            "Motor position commands (12D): target joint angles relative to default pose"
        )

        # Save the updated model
        onnx.save(onnx_model, output_path)

        # Validate the model
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model validation passed")
        print(f"✅ Added {len(metadata)} metadata properties")

    except ImportError:
        print("⚠️  onnx not installed, skipping validation and metadata")
    except Exception as e:
        print(f"⚠️  ONNX validation/metadata failed: {e}")


def export_to_torchscript(model, obs_dim, output_path, std=None):
    """Export the policy network to TorchScript format"""
    print(f"\n📦 Exporting to TorchScript: {output_path}")

    # Set model to eval mode
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, obs_dim)

    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)

    # Save the model with metadata
    if std is not None:
        # Create a wrapper that includes std
        class PolicyWithStd(nn.Module):
            def __init__(self, policy, std):
                super().__init__()
                self.policy = policy
                self.register_buffer('std', std)

            def forward(self, x):
                return self.policy(x)

        full_model = PolicyWithStd(model, std)
        traced_full = torch.jit.trace(full_model, dummy_input)
        torch.jit.save(traced_full, output_path)
    else:
        torch.jit.save(traced_model, output_path)

    print(f"✅ Exported TorchScript model to: {output_path}")

    # Test loading
    loaded_model = torch.jit.load(output_path)
    loaded_model.eval()
    with torch.no_grad():
        test_output = loaded_model(dummy_input)
    print(f"✅ TorchScript test passed, output shape: {test_output.shape}")


def generate_json_documentation(obs_dim, action_dim, hidden_dims, activation, std, iteration, exp_name, robot_type, output_path, control_freq=None):
    """Generate comprehensive JSON documentation for the policy model"""
    print(f"\n📄 Generating JSON documentation: {output_path}")

    # Calculate dt from control_freq if available
    dt = 1.0 / control_freq if control_freq else None

    doc = {
        "model_info": {
            "name": f"{robot_type.upper()} Quadruped Locomotion Policy",
            "robot": robot_type.upper(),
            "framework": "Genesis",
            "experiment_name": exp_name,
            "training_iterations": iteration,
            "model_type": "quadruped_locomotion_policy",
        },
        "architecture": {
            "type": "MLP",
            "input_dimension": obs_dim,
            "output_dimension": action_dim,
            "hidden_dimensions": hidden_dims,
            "activation": activation,
            "layer_structure": f"{obs_dim} → {' → '.join(map(str, hidden_dims))} → {action_dim}",
        },
        "observation_space": {
            "total_dimensions": obs_dim,
            "components": [
                {
                    "name": "Angular Velocity",
                    "indices": "[0:3]",
                    "dimensions": 3,
                    "description": "Base angular velocity in robot frame (roll, pitch, yaw)",
                    "units": "rad/s",
                    "scaling_factor": 0.25,
                },
                {
                    "name": "Projected Gravity",
                    "indices": "[3:6]",
                    "dimensions": 3,
                    "description": "Gravity vector projected into robot frame (for orientation)",
                    "units": "unit vector",
                    "scaling_factor": 1.0,
                },
                {
                    "name": "Command Velocities",
                    "indices": "[6:9]",
                    "dimensions": 3,
                    "description": "Desired velocities: [vx, vy, yaw_rate]",
                    "units": "m/s, rad/s",
                    "scaling_factor": 2.0,
                },
                {
                    "name": "Joint Positions",
                    "indices": "[9:21]",
                    "dimensions": 12,
                    "description": "Current joint angles relative to default pose",
                    "units": "rad",
                    "scaling_factor": 1.0,
                },
                {
                    "name": "Joint Velocities",
                    "indices": "[21:33]",
                    "dimensions": 12,
                    "description": "Current joint angular velocities",
                    "units": "rad/s",
                    "scaling_factor": 0.05,
                },
                {
                    "name": "Previous Actions",
                    "indices": "[33:45]",
                    "dimensions": 12,
                    "description": "Previous motor commands (for temporal consistency)",
                    "units": "rad",
                    "scaling_factor": 1.0,
                },
            ],
        },
        "action_space": {
            "dimensions": action_dim,
            "description": "Motor position commands (rad) relative to default pose",
            "scaling": {"action_scale": 0.25, "formula": "target_pos = action * 0.25 + default_pose"},
            "std_values": std.tolist() if std is not None else None,
        },
        "joint_configuration": {
            "joint_order": [
                "FR_hip",
                "FR_thigh",
                "FR_calf",
                "FL_hip",
                "FL_thigh",
                "FL_calf",
                "RR_hip",
                "RR_thigh",
                "RR_calf",
                "RL_hip",
                "RL_thigh",
                "RL_calf",
            ],
            "default_angles": {
                "description": "Default joint angles in radians",
                "hip_joints": {"FR": 0.0, "FL": 0.0, "RR": 0.0, "RL": 0.0},
                "thigh_joints": {"FR": 0.8, "FL": 0.8, "RR": 1.0, "RL": 1.0},
                "calf_joints": {"FR": -1.5, "FL": -1.5, "RR": -1.5, "RL": -1.5},
            },
        },
        "control_parameters": {
            "pd_gains": {"kp": 20.0, "kd": 0.5},
            "control_frequency": f"{control_freq}Hz" if control_freq else "Configurable",
            "timestep": dt if dt else "Configurable",
            "action_latency": "1 step (simulated)",
        },
        "usage": {
            "onnx": {
                "load": "onnxruntime.InferenceSession('model.onnx')",
                "input_name": "observation",
                "output_name": "action_mean",
                "input_shape": [1, obs_dim],
                "output_shape": [1, action_dim],
            },
            "torchscript": {
                "load": "torch.jit.load('model.pt')",
                "input_shape": [1, obs_dim],
                "output_shape": [1, action_dim],
                "has_embedded_std": True,
            },
        },
    }

    # Save JSON documentation
    with open(output_path, 'w') as f:
        json.dump(doc, f, indent=2)

    print(f"✅ Generated JSON documentation with {len(doc)} main sections")
    return doc


def main():
    parser = argparse.ArgumentParser(description="Extract Go1/Go2 policy to ONNX/TorchScript")
    parser.add_argument("-r", "--robot", type=str, default="go1", choices=["go1", "go2"], help="Robot type to extract")
    parser.add_argument("-e", "--exp_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--ckpt", type=int, default=1400, help="Checkpoint number")
    parser.add_argument(
        "--format", type=str, default="both", choices=["onnx", "torchscript", "both"], help="Export format"
    )
    parser.add_argument("--output_dir", type=str, default="./exported_models", help="Output directory")
    args = parser.parse_args()

    # Set default experiment name based on robot type if not provided
    if args.exp_name is None:
        args.exp_name = f"{args.robot}-locomotion-train"

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"{args.robot.upper()} Policy Network Extractor (Standalone)")
    print("=" * 60)
    print(f"Robot: {args.robot.upper()}")
    print(f"Experiment: {args.exp_name}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Export format: {args.format}")
    print("=" * 60)

    # Load configs to get dimensions
    cfg_path = f"logs/{args.exp_name}/cfgs.pkl"
    if not os.path.exists(cfg_path):
        print(f"❌ Config file not found: {cfg_path}")
        return

    print(f"\n📄 Loading configs from: {cfg_path}")
    with open(cfg_path, 'rb') as f:
        cfg_data = pickle.load(f)

    env_cfg = cfg_data['task_config']['env_cfg']
    obs_cfg = cfg_data['task_config']['obs_cfg']
    train_cfg = cfg_data['rsl_rl_config'].policy

    # Extract robot config to get control frequency
    # The frequency is stored in robot_cfgs list at the top level
    control_freq = None
    if 'robot_cfgs' in cfg_data and len(cfg_data['robot_cfgs']) > 0:
        # Get frequency from first robot config
        control_freq = cfg_data['robot_cfgs'][0].get('frequency', None)

    # Extract dimensions
    obs_dim = obs_cfg['num_obs']
    action_dim = env_cfg['num_actions']

    hidden_dims = train_cfg.actor_hidden_dims
    activation = train_cfg.activation

    print(f"  - Observation dimension: {obs_dim}")
    print(f"  - Action dimension: {action_dim}")
    print(f"  - Hidden dimensions: {hidden_dims}")
    if control_freq:
        print(f"  - Control frequency: {control_freq}Hz (dt={1.0/control_freq:.3f}s)")

    # Create policy network
    policy = create_policy_network(obs_dim, action_dim, hidden_dims, activation)

    # Load checkpoint weights
    checkpoint_path = f"logs/{args.exp_name}/model_{args.ckpt}.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    actor_weights, std, iteration = extract_actor_weights(checkpoint_path)
    print(f"  - Checkpoint iteration: {iteration}")

    # Load weights into policy network
    policy.net.load_state_dict(actor_weights)
    print("✅ Loaded actor weights into policy network")

    # Export to desired formats
    base_name = f"{args.exp_name}_policy_iter{iteration}"

    if args.format in ["onnx", "both"]:
        onnx_path = os.path.join(args.output_dir, f"{base_name}.onnx")
        export_to_onnx(policy, obs_dim, onnx_path, args.robot, std, control_freq)

    if args.format in ["torchscript", "both"]:
        ts_path = os.path.join(args.output_dir, f"{base_name}.pt")
        export_to_torchscript(policy, obs_dim, ts_path, std)

    # Generate JSON documentation
    json_path = os.path.join(args.output_dir, f"{base_name}.json")
    generate_json_documentation(obs_dim, action_dim, hidden_dims, activation, std, iteration, args.exp_name, args.robot, json_path, control_freq)

    # Print usage instructions
    print("\n" + "=" * 60)
    print("🎉 Export complete!")
    print("=" * 60)
    # print("\n📝 Usage examples:")

    # if args.format in ["onnx", "both"]:
    #     print("\n### Reading ONNX Model Documentation:")
    #     print("```python")
    #     print("import onnx")
    #     print(f"model = onnx.load('{base_name}.onnx')")
    #     print("# Print comprehensive documentation")
    #     print("print(model.doc_string)")
    #     print("# Print metadata properties")
    #     print("for prop in model.metadata_props:")
    #     print("    print(f'{prop.key}: {prop.value}')")
    #     print("```")
    #     print("")
    #     print("### Python ONNX inference:")
    #     print("```python")
    #     print("import onnxruntime as ort")
    #     print("import numpy as np")
    #     print("")
    #     print(f"# Load model")
    #     print(f"session = ort.InferenceSession('{base_name}.onnx')")
    #     print("")
    #     print(f"# Create observation (shape: {obs_dim})")
    #     print("# Components: [ang_vel(3), gravity(3), commands(3), joint_pos(12), joint_vel(12), prev_actions(12)]")
    #     print(f"obs = np.random.randn(1, {obs_dim}).astype(np.float32)")
    #     print("")
    #     print("# Run inference")
    #     print("action_mean = session.run(None, {'observation': obs})[0]")
    #     print(f"# Output shape: (1, {action_dim}) - motor position commands")
    #     print("```")

    # if args.format in ["torchscript", "both"]:
    #     print("\n### Python TorchScript inference:")
    #     print("```python")
    #     print("import torch")
    #     print("")
    #     print(f"# Load model")
    #     print(f"model = torch.jit.load('{base_name}.pt')")
    #     print("model.eval()")
    #     print("")
    #     print(f"# Create observation (shape: {obs_dim})")
    #     print(f"obs = torch.randn(1, {obs_dim})")
    #     print("")
    #     print("# Run inference")
    #     print("with torch.no_grad():")
    #     print("    action_mean = model(obs)")
    #     if std is not None:
    #         print("    action_std = model.std  # If using PolicyWithStd wrapper")
    #     print(f"# Output shape: (1, {action_dim})")
    #     print("```")


if __name__ == "__main__":
    main()
