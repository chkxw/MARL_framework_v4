#!/usr/bin/env python3
"""
URDF Joint Detector - Extract joint information from URDF files
No external dependencies required - uses only Python standard library
"""

import xml.etree.ElementTree as ET
import glob
import os
from typing import List, Dict, Optional


class URDFJointDetector:
    """Extract joint information from URDF files without external dependencies."""

    def __init__(self):
        self.joints = []
        self.links = []

    def parse_urdf(self, urdf_file: str) -> List[Dict[str, str]]:
        """
        Parse a URDF file and extract joint information.

        Args:
            urdf_file: Path to the URDF file

        Returns:
            List of dictionaries containing joint information
        """
        try:
            tree = ET.parse(urdf_file)
            root = tree.getroot()

            # Extract joints
            self.joints = []
            for joint in root.findall('joint'):
                joint_info = {'name': joint.get('name', 'unnamed'), 'type': joint.get('type', 'unknown')}

                # Get parent and child links
                parent = joint.find('parent')
                child = joint.find('child')
                if parent is not None:
                    joint_info['parent_link'] = parent.get('link', 'unknown')
                if child is not None:
                    joint_info['child_link'] = child.get('link', 'unknown')

                # Get joint limits if available
                limit = joint.find('limit')
                if limit is not None:
                    joint_info['lower_limit'] = limit.get('lower', 'N/A')
                    joint_info['upper_limit'] = limit.get('upper', 'N/A')
                    joint_info['effort'] = limit.get('effort', 'N/A')
                    joint_info['velocity'] = limit.get('velocity', 'N/A')

                # Get axis if available
                axis = joint.find('axis')
                if axis is not None:
                    joint_info['axis'] = axis.get('xyz', '1 0 0')

                self.joints.append(joint_info)

            # Extract links for reference
            self.links = [link.get('name', 'unnamed') for link in root.findall('link')]

            return self.joints

        except ET.ParseError as e:
            print(f"Error parsing XML in {urdf_file}: {e}")
            return []
        except FileNotFoundError:
            print(f"File not found: {urdf_file}")
            return []
        except Exception as e:
            print(f"Unexpected error processing {urdf_file}: {e}")
            return []

    def get_joint_names(self) -> List[str]:
        """Get list of joint names in order."""
        return [joint['name'] for joint in self.joints]

    def get_movable_joints(self) -> List[Dict[str, str]]:
        """Get only movable joints (revolute, prismatic, continuous)."""
        movable_types = ['revolute', 'prismatic', 'continuous']
        return [joint for joint in self.joints if joint['type'] in movable_types]

    def print_joint_summary(self, urdf_file: str):
        """Print a formatted summary of joints in the URDF."""
        print(f"\n{'='*60}")
        print(f"URDF File: {os.path.basename(urdf_file)}")
        print(f"{'='*60}")

        if not self.joints:
            print("No joints found in this URDF file.")
            return

        print(f"\nTotal joints: {len(self.joints)}")
        print(f"Total links: {len(self.links)}")

        # Count joint types
        joint_types = {}
        for joint in self.joints:
            jtype = joint['type']
            joint_types[jtype] = joint_types.get(jtype, 0) + 1

        print("\nJoint types:")
        for jtype, count in joint_types.items():
            print(f"  - {jtype}: {count}")

        print("\nJoints in order:")
        for i, joint in enumerate(self.joints, 1):
            print(f"\n  {i}. {joint['name']}")
            print(f"     Type: {joint['type']}")
            if 'parent_link' in joint:
                print(f"     Parent: {joint['parent_link']}")
            if 'child_link' in joint:
                print(f"     Child: {joint['child_link']}")
            if 'axis' in joint:
                print(f"     Axis: {joint['axis']}")
            if joint['type'] in ['revolute', 'prismatic']:
                if 'lower_limit' in joint:
                    print(f"     Limits: [{joint['lower_limit']}, {joint['upper_limit']}]")





if __name__ == "__main__":
    """Example: Process a single URDF file."""
    detector = URDFJointDetector()

    # Replace with your actual URDF file path
    urdf_file = "/home/yuhao2024/Documents/Genesis_latest/genesis/assets/urdf/go2/urdf/go2.urdf"

    joints = detector.parse_urdf(urdf_file)

    # Get just the names
    joint_names = detector.get_joint_names()
    print("Joint names in order:", joint_names)

    # Get only movable joints
    movable = detector.get_movable_joints()
    print("\nMovable joints:")
    for joint in movable:
        print(f"  - {joint['name']} ({joint['type']})")

