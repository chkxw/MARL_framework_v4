#!/usr/bin/env python3
"""
Extract actor network from OpenRL PPO checkpoint for evaluation.

This script loads an OpenRL checkpoint file and extracts just the actor network,
saving it in both PyTorch (.pt) and ONNX (.onnx) formats with comprehensive documentation.
"""

import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import pickle
import json
import numpy as np
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnx/onnxruntime not installed. ONNX export will be limited.")


def extract_openrl_actor(checkpoint_path: str, output_path: str = None, verbose: bool = True):
    """Extract actor network from OpenRL checkpoint.
    
    Args:
        checkpoint_path: Path to OpenRL checkpoint (.pt file or directory containing module.pt)
        output_path: Output path for extracted actor network (defaults to checkpoint_path with _actor suffix)
        verbose: Print extraction details
        
    Returns:
        Path to saved actor network
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Handle both file and directory inputs
    if checkpoint_path.is_dir():
        # OpenRL saves checkpoints in directories with module.pt files
        module_path = checkpoint_path / "module.pt"
        if module_path.exists():
            checkpoint_path = module_path
            if verbose:
                print(f"Found module.pt in directory: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"No module.pt found in directory: {checkpoint_path}")
    elif not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if verbose:
        print(f"Loading OpenRL checkpoint from: {checkpoint_path}")
    
    # Load the checkpoint (OpenRL checkpoints may contain custom classes)
    try:
        # Try loading with weights_only=False since OpenRL checkpoints contain PPOModule objects
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        if verbose:
            print(f"Warning: Failed to load with weights_only=False, trying alternative: {e}")
        # Fallback to regular load
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if verbose:
        print(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}")
    
    # Handle different checkpoint formats
    model = None
    
    # Check if it's an OpenRL PPOModule
    if hasattr(checkpoint, 'models'):
        # OpenRL PPOModule format
        if verbose:
            print(f"Found OpenRL PPOModule with models: {list(checkpoint.models.keys())}")
        
        # Get the PolicyValueNetwork
        if 'model' in checkpoint.models:
            pv_network = checkpoint.models['model']
            if hasattr(pv_network, 'state_dict'):
                model = pv_network.state_dict()
                if verbose:
                    print(f"Extracted state_dict from PolicyValueNetwork")
        else:
            # Try to get state dict from the first model
            for name, net in checkpoint.models.items():
                if hasattr(net, 'state_dict'):
                    model = net.state_dict()
                    if verbose:
                        print(f"Extracted state_dict from {name}")
                    break
    
    elif isinstance(checkpoint, dict):
        # Standard checkpoint format
        if 'model' in checkpoint:
            model = checkpoint['model']
        elif 'state_dict' in checkpoint:
            model = checkpoint['state_dict']
        elif 'actor' in checkpoint:
            model = checkpoint['actor']
        else:
            # Assume the checkpoint itself is the state dict
            model = checkpoint
            
        if verbose:
            if isinstance(model, dict):
                print(f"Model state dict keys: {list(model.keys())[:10]}...")  # Show first 10 keys
            else:
                print(f"Model type: {type(model)}")
    else:
        # The checkpoint might be the model itself
        if hasattr(checkpoint, 'state_dict'):
            model = checkpoint.state_dict()
        else:
            model = checkpoint
        if verbose:
            print(f"Checkpoint is directly a model of type: {type(checkpoint)}")
    
    # Extract actor-specific weights
    actor_weights = {}
    
    if isinstance(model, dict):
        # Check for OpenRL PolicyValueNetwork structure
        has_openrl_structure = any('obs_prep' in k or 'act.' in k for k in model.keys())
        has_critic_components = any('critic' in k.lower() or 'v_out' in k for k in model.keys())
        
        if has_openrl_structure and has_critic_components:
            # This is an OpenRL PolicyValueNetwork
            # Actor components: obs_prep, common, act
            # Critic components: critic_obs_prep, v_out
            for key, value in model.items():
                # Include only actor components
                if any(comp in key for comp in ['obs_prep.', 'common.', 'act.']):
                    # Exclude critic_obs_prep
                    if 'critic_obs_prep' not in key:
                        actor_weights[key] = value
            if verbose:
                print(f"Extracted {len(actor_weights)} actor weights from OpenRL PolicyValueNetwork")
                print(f"  Included: obs_prep, common, act")
                print(f"  Excluded: critic_obs_prep, v_out")
        
        elif any(k.startswith('actor.') for k in model.keys()):
            # Standard actor. prefix format
            for key, value in model.items():
                if key.startswith('actor.'):
                    new_key = key[6:]  # Remove 'actor.' prefix
                    actor_weights[new_key] = value
            if verbose:
                print(f"Extracted {len(actor_weights)} actor weights with 'actor.' prefix")
        
        else:
            # Fallback: filter out critic components
            for key, value in model.items():
                # Skip critic/value related weights
                if not any(x in key.lower() for x in ['critic', 'value', 'v_out', 'v_']):
                    actor_weights[key] = value
            if verbose:
                if actor_weights:
                    print(f"Extracted {len(actor_weights)} weights (filtered out critic components)")
                else:
                    # If filtering leaves nothing, use all weights
                    actor_weights = model
                    print(f"Warning: Could not identify actor components, using all {len(actor_weights)} weights")
    
    if not actor_weights:
        raise ValueError("Could not extract actor weights from checkpoint")
    
    # Determine output path
    if output_path is None:
        # If checkpoint was a directory, use the directory name for output
        if checkpoint_path.name == "module.pt":
            parent_dir = checkpoint_path.parent
            output_path = parent_dir.parent / f"{parent_dir.name}_actor.pt"
        else:
            output_path = checkpoint_path.parent / f"{checkpoint_path.stem}_actor.pt"
    else:
        output_path = Path(output_path)
    
    # Save the extracted actor network
    torch.save(actor_weights, output_path)
    
    if verbose:
        print(f"✅ Saved extracted actor network to: {output_path}")
        print(f"   Number of parameters: {len(actor_weights)}")
        print(f"   Total parameter count: {sum(p.numel() for p in actor_weights.values() if isinstance(p, torch.Tensor)):,}")
        
        # Show first few layer names for verification
        layer_names = list(actor_weights.keys())[:5]
        print(f"   First few layers: {layer_names}")
    
    return str(output_path)


def reconstruct_openrl_actor_network(state_dict, verbose=True):
    """Reconstruct the actor network architecture from OpenRL state dict.
    
    This recreates the network structure used by OpenRL's PolicyValueNetwork
    for the actor components (obs_prep, common, act).
    """
    
    # Analyze the state dict to determine network dimensions
    input_dim = None
    hidden_dim = None
    output_dim = None
    
    # Find dimensions from the state dict keys
    for key, tensor in state_dict.items():
        if key == 'obs_prep.mlp.fc1.0.weight':
            hidden_dim, input_dim = tensor.shape
        elif key == 'act.action_out.fc_mean.weight':
            output_dim, _ = tensor.shape
            
    if verbose:
        print(f"\nNetwork dimensions detected:")
        print(f"  Input: {input_dim}")
        print(f"  Hidden: {hidden_dim}")
        print(f"  Output: {output_dim}")
    
    # Build a simplified actor network that matches the forward pass
    class OpenRLActor(nn.Module):
        """Simplified OpenRL actor network for inference"""
        def __init__(self):
            super().__init__()
            # Observation preprocessing
            self.obs_norm = nn.LayerNorm(input_dim)
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc1_activation = nn.ELU()
            self.fc1_norm = nn.LayerNorm(hidden_dim)
            
            self.fc_h = nn.Linear(hidden_dim, hidden_dim)
            self.fc_h_activation = nn.ELU()
            self.fc_h_norm = nn.LayerNorm(hidden_dim)
            
            # Two parallel streams (from fc2.0 and fc2.1)
            self.fc2_0 = nn.Linear(hidden_dim, hidden_dim)
            self.fc2_0_activation = nn.ELU()
            self.fc2_0_norm = nn.LayerNorm(hidden_dim)
            
            self.fc2_1 = nn.Linear(hidden_dim, hidden_dim)
            self.fc2_1_activation = nn.ELU()
            self.fc2_1_norm = nn.LayerNorm(hidden_dim)
            
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3_norm = nn.LayerNorm(hidden_dim)
            
            # Common layers
            self.common_fc1 = nn.Linear(hidden_dim, hidden_dim)
            self.common_fc1_activation = nn.ELU()
            self.common_fc1_norm = nn.LayerNorm(hidden_dim)
            
            self.common_fc3 = nn.Linear(hidden_dim, hidden_dim)
            self.common_fc3_norm = nn.LayerNorm(hidden_dim)
            
            # Action output
            self.action_mean = nn.Linear(hidden_dim, output_dim)
            self.action_logstd = nn.Parameter(torch.zeros(output_dim))
            
        def forward(self, obs):
            # Observation preprocessing
            x = self.obs_norm(obs)
            x = self.fc1(x)
            x = self.fc1_activation(x)
            x = self.fc1_norm(x)
            
            x = self.fc_h(x)
            x = self.fc_h_activation(x)
            x = self.fc_h_norm(x)
            
            # Parallel streams
            x0 = self.fc2_0(x)
            x0 = self.fc2_0_activation(x0)
            x0 = self.fc2_0_norm(x0)
            
            x1 = self.fc2_1(x)
            x1 = self.fc2_1_activation(x1)
            x1 = self.fc2_1_norm(x1)
            
            # Combine streams (average)
            x = (x0 + x1) / 2.0
            
            x = self.fc3(x)
            x = self.fc3_norm(x)
            
            # Common layers
            x = self.common_fc1(x)
            x = self.common_fc1_activation(x)
            x = self.common_fc1_norm(x)
            
            x = self.common_fc3(x)
            x = self.common_fc3_norm(x)
            
            # Action output
            action_mean = self.action_mean(x)
            
            return action_mean
    
    # Create the network
    actor = OpenRLActor()
    
    # Load the weights with proper mapping
    new_state_dict = {}
    for key, value in state_dict.items():
        # Map OpenRL keys to our simplified network
        if key == 'obs_prep.feature_norm.weight':
            new_state_dict['obs_norm.weight'] = value
        elif key == 'obs_prep.feature_norm.bias':
            new_state_dict['obs_norm.bias'] = value
        elif key == 'obs_prep.mlp.fc1.0.weight':
            new_state_dict['fc1.weight'] = value
        elif key == 'obs_prep.mlp.fc1.0.bias':
            new_state_dict['fc1.bias'] = value
        elif key == 'obs_prep.mlp.fc1.2.weight':
            new_state_dict['fc1_norm.weight'] = value
        elif key == 'obs_prep.mlp.fc1.2.bias':
            new_state_dict['fc1_norm.bias'] = value
        elif key == 'obs_prep.mlp.fc_h.0.weight':
            new_state_dict['fc_h.weight'] = value
        elif key == 'obs_prep.mlp.fc_h.0.bias':
            new_state_dict['fc_h.bias'] = value
        elif key == 'obs_prep.mlp.fc_h.2.weight':
            new_state_dict['fc_h_norm.weight'] = value
        elif key == 'obs_prep.mlp.fc_h.2.bias':
            new_state_dict['fc_h_norm.bias'] = value
        elif key == 'obs_prep.mlp.fc2.0.0.weight':
            new_state_dict['fc2_0.weight'] = value
        elif key == 'obs_prep.mlp.fc2.0.0.bias':
            new_state_dict['fc2_0.bias'] = value
        elif key == 'obs_prep.mlp.fc2.0.2.weight':
            new_state_dict['fc2_0_norm.weight'] = value
        elif key == 'obs_prep.mlp.fc2.0.2.bias':
            new_state_dict['fc2_0_norm.bias'] = value
        elif key == 'obs_prep.mlp.fc2.1.0.weight':
            new_state_dict['fc2_1.weight'] = value
        elif key == 'obs_prep.mlp.fc2.1.0.bias':
            new_state_dict['fc2_1.bias'] = value
        elif key == 'obs_prep.mlp.fc2.1.2.weight':
            new_state_dict['fc2_1_norm.weight'] = value
        elif key == 'obs_prep.mlp.fc2.1.2.bias':
            new_state_dict['fc2_1_norm.bias'] = value
        elif key == 'obs_prep.mlp.fc3.0.weight':
            new_state_dict['fc3.weight'] = value
        elif key == 'obs_prep.mlp.fc3.0.bias':
            new_state_dict['fc3.bias'] = value
        elif key == 'obs_prep.mlp.fc3.1.weight':
            new_state_dict['fc3_norm.weight'] = value
        elif key == 'obs_prep.mlp.fc3.1.bias':
            new_state_dict['fc3_norm.bias'] = value
        elif key == 'common.fc1.0.weight':
            new_state_dict['common_fc1.weight'] = value
        elif key == 'common.fc1.0.bias':
            new_state_dict['common_fc1.bias'] = value
        elif key == 'common.fc1.2.weight':
            new_state_dict['common_fc1_norm.weight'] = value
        elif key == 'common.fc1.2.bias':
            new_state_dict['common_fc1_norm.bias'] = value
        elif key == 'common.fc3.0.weight':
            new_state_dict['common_fc3.weight'] = value
        elif key == 'common.fc3.0.bias':
            new_state_dict['common_fc3.bias'] = value
        elif key == 'common.fc3.1.weight':
            new_state_dict['common_fc3_norm.weight'] = value
        elif key == 'common.fc3.1.bias':
            new_state_dict['common_fc3_norm.bias'] = value
        elif key == 'act.action_out.fc_mean.weight':
            new_state_dict['action_mean.weight'] = value
        elif key == 'act.action_out.fc_mean.bias':
            new_state_dict['action_mean.bias'] = value
        elif key == 'act.action_out.logstd._bias':
            # Handle shape mismatch - OpenRL may have shape [3, 1], we need [3]
            if value.dim() > 1:
                new_state_dict['action_logstd'] = value.squeeze()
            else:
                new_state_dict['action_logstd'] = value
    
    actor.load_state_dict(new_state_dict)
    actor.eval()
    
    return actor, input_dim, hidden_dim, output_dim


def export_to_onnx_with_docs(actor_weights, onnx_path, verbose=True):
    """Export actor network to ONNX format with comprehensive documentation."""
    
    # Reconstruct the network
    actor, input_dim, hidden_dim, output_dim = reconstruct_openrl_actor_network(actor_weights, verbose)
    
    if verbose:
        print(f"\n📦 Exporting to ONNX: {onnx_path}")
    
    # Create comprehensive documentation
    doc_string = f"""Go2 Element Push MAPPO Shared Actor Network

NETWORK ARCHITECTURE:
- Input dimension: {input_dim} (single robot observation)
- Hidden dimension: {hidden_dim}
- Output dimension: {output_dim} (single robot actions)
- Architecture: OpenRL PolicyValueNetwork (actor components only)
- Training mode: SHARED (same network used by both robots with shared parameters)

OBSERVATION SPACE ({input_dim} dimensions):
┌────────────────────────────────────────────────────────────────────┐
│ Component                      │ Dim │ Description                 │
├────────────────────────────────────────────────────────────────────┤
│ Base Linear Velocity (scaled) │  2  │ XY velocity in robot frame  │
│ Base Angular Velocity (scaled)│  3  │ Angular velocity in robot   │
│ Element Direction (robot fr.)  │  2  │ Element relative position   │
│ Other Robot Direction (robot) │  2  │ Other robot relative pos    │
│ Goal from Robot (robot frame) │  2  │ Goal position from robot    │
│ Goal from Element (world fr.) │  2  │ Goal position from element  │
│ Previous Action               │  3  │ Last velocity commands      │
│ Robot Index                   │  1  │ Agent identification (0/1)  │
└────────────────────────────────────────────────────────────────────┘

ACTION SPACE ({output_dim} dimensions):
- Forward velocity command (m/s)
- Lateral velocity command (m/s)
- Angular velocity command (rad/s)

TRAINING:
- Algorithm: MAPPO (Multi-Agent PPO)
- Framework: OpenRL
- Training mode: SHARED (parameter sharing between agents)
- Environment: Dual Go2 robots collaborative element pushing task

USAGE:
This is a shared-parameter network. Each robot uses the same network independently:
- Input: Single robot's observation ({input_dim}D)
- Output: Single robot's action ({output_dim}D)
- Both robots use identical network weights
- The robot_idx in observation allows network to distinguish between agents
"""
    
    # Create dummy input for tracing
    dummy_input = torch.randn(1, input_dim)
    
    # Test forward pass
    with torch.no_grad():
        test_output = actor(dummy_input)
        if verbose:
            print(f"  Test output shape: {test_output.shape}")
    
    # Export to ONNX
    torch.onnx.export(
        actor,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        },
        verbose=False
    )
    
    if verbose:
        print(f"✅ Exported ONNX model to: {onnx_path}")
    
    # Add metadata if ONNX is available
    if ONNX_AVAILABLE:
        try:
            # Load the model
            onnx_model = onnx.load(onnx_path)
            
            # Add documentation
            onnx_model.doc_string = doc_string
            
            # Add metadata properties
            metadata = [
                ("model_type", "multi_agent_shared_locomotion_policy"),
                ("robots", "go2"),
                ("algorithm", "MAPPO"),
                ("framework", "OpenRL"),
                ("task", "collaborative_element_pushing"),
                ("training_mode", "SHARED"),
                ("observation_dim", str(input_dim)),
                ("action_dim", str(output_dim)),
                ("hidden_dim", str(hidden_dim)),
                ("num_agents", "2"),
            ]

            for key, value in metadata:
                onnx_model.metadata_props.append(
                    onnx.StringStringEntryProto(key=key, value=value)
                )

            # Add input/output descriptions
            onnx_model.graph.input[0].doc_string = (
                f"Single robot observation ({input_dim}D): "
                "[lin_vel(2), ang_vel(3), element_dir(2), other_robot_dir(2), goal_from_robot(2), goal_from_element(2), prev_action(3), robot_idx(1)]"
            )
            onnx_model.graph.output[0].doc_string = (
                f"Single robot action ({output_dim}D): "
                "[forward_vel, lateral_vel, angular_vel]"
            )
            
            # Save updated model
            onnx.save(onnx_model, onnx_path)
            
            # Validate
            onnx.checker.check_model(onnx_model)
            if verbose:
                print(f"✅ Added documentation and {len(metadata)} metadata properties")
                print("✅ ONNX model validation passed")
                
        except Exception as e:
            if verbose:
                print(f"⚠️ Could not add ONNX metadata: {e}")
    
    return input_dim, hidden_dim, output_dim


def generate_json_documentation(actor_weights, json_path, checkpoint_info, verbose=True):
    """Generate comprehensive JSON documentation for the extracted model."""
    
    if verbose:
        print(f"\n📄 Generating JSON documentation: {json_path}")
    
    # Get network dimensions
    input_dim = None
    output_dim = None
    for key, tensor in actor_weights.items():
        if key == 'obs_prep.mlp.fc1.0.weight':
            _, input_dim = tensor.shape
        elif key == 'act.action_out.fc_mean.weight':
            output_dim, _ = tensor.shape
    
    doc = {
        "model_info": {
            "name": "Go2 Element Push MAPPO Shared Actor",
            "type": "multi_agent_shared_locomotion_policy",
            "algorithm": "MAPPO",
            "framework": "OpenRL",
            "training_mode": "SHARED",
            "task": "collaborative_element_pushing",
            "checkpoint_path": checkpoint_info.get("path", "unknown"),
            "extraction_date": str(Path(json_path).stat().st_mtime if Path(json_path).exists() else "new"),
        },
        "architecture": {
            "type": "PolicyValueNetwork_Actor",
            "components": ["obs_prep", "common", "act"],
            "input_dimension": input_dim,
            "output_dimension": output_dim,
            "hidden_dimension": 128,
            "activation": "ELU",
            "normalization": "LayerNorm",
        },
        "observation_space": {
            "total_dimensions": input_dim,
            "num_agents": 2,
            "observation_components": [
                {"name": "Base Linear Velocity (scaled)", "dimensions": 2, "indices": "[0:2]"},
                {"name": "Base Angular Velocity (scaled)", "dimensions": 3, "indices": "[2:5]"},
                {"name": "Element Direction (robot frame)", "dimensions": 2, "indices": "[5:7]"},
                {"name": "Other Robot Direction (robot frame)", "dimensions": 2, "indices": "[7:9]"},
                {"name": "Goal from Robot (robot frame)", "dimensions": 2, "indices": "[9:11]"},
                {"name": "Goal from Element (world frame)", "dimensions": 2, "indices": "[11:13]"},
                {"name": "Previous Action", "dimensions": 3, "indices": "[13:16]"},
                {"name": "Robot Index", "dimensions": 1, "indices": "[16:17]"},
            ],
        },
        "action_space": {
            "total_dimensions": output_dim,
            "action_components": [
                {"name": "Forward Velocity", "dimension": 1, "units": "m/s"},
                {"name": "Lateral Velocity", "dimension": 1, "units": "m/s"},
                {"name": "Angular Velocity", "dimension": 1, "units": "rad/s"},
            ],
        },
        "usage": {
            "pytorch": {
                "load": f"actor_weights = torch.load('{Path(json_path).stem}.pt')",
                "inference": "Use with ModelInference class or load into network structure",
            },
            "onnx": {
                "load": f"session = ort.InferenceSession('{Path(json_path).stem}.onnx')",
                "input_name": "observation",
                "output_name": "action",
                "input_shape": f"[batch_size, {input_dim}]",
                "output_shape": f"[batch_size, {output_dim}]",
            }
        },
        "notes": [
            "This is the actor network only (critic components removed)",
            "SHARED training mode: same network weights used by both agents",
            "Each robot uses this network independently with its own observation",
            "Network processes single robot observation (18D) and outputs single robot action (3D)",
            "Robot index in observation allows network to distinguish between agents",
            "Suitable for evaluation/deployment",
        ]
    }
    
    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(doc, f, indent=2)
    
    if verbose:
        print(f"✅ Generated JSON documentation with {len(doc)} sections")
    
    return doc


def main():
    parser = argparse.ArgumentParser(description="Extract actor network from OpenRL checkpoint")
    parser.add_argument(
        "checkpoint", 
        type=str, 
        help="Path to OpenRL checkpoint file (.pt) or directory containing module.pt"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output base path (will generate .pt, .onnx, and .json files)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="both",
        choices=["pt", "onnx", "both"],
        help="Export format: pt (PyTorch), onnx, or both (default: both)"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    try:
        # First extract the actor weights to .pt format
        pt_output_path = extract_openrl_actor(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            verbose=verbose
        )
        
        # Load the extracted weights
        actor_weights = torch.load(pt_output_path, map_location='cpu')
        
        # Prepare paths
        base_path = Path(pt_output_path).with_suffix('')
        
        # Export to ONNX if requested
        if args.format in ["onnx", "both"]:
            onnx_path = str(base_path) + ".onnx"
            export_to_onnx_with_docs(actor_weights, onnx_path, verbose)
        
        # Generate JSON documentation
        json_path = str(base_path) + ".json"
        checkpoint_info = {"path": args.checkpoint}
        generate_json_documentation(actor_weights, json_path, checkpoint_info, verbose)
        
        # Print usage instructions
        if verbose:
            print("\n" + "=" * 60)
            print("🎉 Extraction complete!")
            print("=" * 60)
            print("\nGenerated files:")
            print(f"  📦 PyTorch weights: {pt_output_path}")
            if args.format in ["onnx", "both"]:
                print(f"  📦 ONNX model: {base_path}.onnx")
            print(f"  📄 Documentation: {base_path}.json")
            
            print("\n📝 Usage examples:")
            
            print("\n### For evaluation with ModelInference:")
            print("```python")
            print("from model_inference import ModelInference")
            print("inference = ModelInference()")
            print(f"inference.load_model('{pt_output_path}')")
            print("# Or use ONNX:")
            if args.format in ["onnx", "both"]:
                print(f"inference.load_model('{base_path}.onnx')")
            print("```")
            
            if args.format in ["onnx", "both"]:
                print("\n### Direct ONNX inference:")
                print("```python")
                print("import onnxruntime as ort")
                print("import numpy as np")
                print(f"session = ort.InferenceSession('{base_path}.onnx')")
                print("obs = np.random.randn(1, 18).astype(np.float32)  # Single robot obs (18D)")
                print("actions = session.run(None, {'observation': obs})[0]  # Single robot action (3D)")
                print("# For both robots, run inference twice with each robot's observation")
                print("```")
                
                print("\n### Reading ONNX documentation:")
                print("```python")
                print("import onnx")
                print(f"model = onnx.load('{base_path}.onnx')")
                print("print(model.doc_string)  # Full documentation")
                print("for prop in model.metadata_props:")
                print("    print(f'{prop.key}: {prop.value}')")
                print("```")
            
            print("\n### Use in evaluation script:")
            print("```bash")
            print(f"python eval_go2_dual_element_push_mappo.py {pt_output_path} \\")
            print("    --num_envs 16 --render --enable_camera")
            print("```")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()