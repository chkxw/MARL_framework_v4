# NAO Robot Standalone Package for Genesis

This folder contains everything needed to load and control the NAO robot in Genesis simulator.

## Structure

```
nao_robot_standalone/
├── urdf/
│   ├── nao_original.urdf    # Original URDF with package:// references
│   ├── nao_local.urdf       # Modified URDF with local mesh paths
│   └── nao_visual_only.urdf # Simplified URDF (visual meshes only)
├── meshes/
│   └── V40/                 # NAO V4.0 mesh files
│       ├── *.dae           # Visual mesh files (39 files)
│       └── *.stl           # Collision mesh files (39 files)
└── textures/
    └── textureNAO.png      # NAO texture file
```

## Usage

### Quick Test
```bash
# From Genesis root directory
python generic_interactive_explorer.py
# Select "nao" from the menu
```

### Custom Script
```python
import genesis as gs

# Initialize Genesis
gs.init()

# Create scene
scene = gs.Scene()
scene.add_entity(gs.morphs.Plane())

# Load NAO robot
nao = scene.add_entity(
    gs.morphs.URDF(
        file="nao_robot_standalone/urdf/nao_local.urdf",
        pos=[0.0, 0.0, 0.35],  # NAO is ~57cm tall
        quat=[0.0, 0.0, 0.0, 1.0]
    )
)

# Build and run
scene.build()
```

## Mesh Files

The package includes both visual (.dae) and collision (.stl) meshes for all NAO body parts:
- Head (HeadYaw, HeadPitch)
- Arms (Shoulder, Elbow, Wrist joints)
- Legs (Hip, Knee, Ankle joints)
- Torso
- Hands (including individual finger segments)

## License

The NAO meshes are licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Public License by Aldebaran/SoftBank Robotics.