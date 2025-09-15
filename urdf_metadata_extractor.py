#!/usr/bin/env python3
"""
URDF Metadata Extractor - Extract joint information and bounding boxes from URDF files
Enhanced version with bounding box calculation - uses only Python standard library
"""

import math
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Union


class URDFMetadataExtractor:
    """Extract joint information and bounding boxes from URDF files without external dependencies."""

    def __init__(self):
        self.joints = []
        self.links = []
        self.link_geometries = {} 
        self.link_bounding_boxes = {} 

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

            # Extract joints (original functionality)
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

            # Extract links for reference (original functionality)
            self.links = [link.get('name', 'unnamed') for link in root.findall('link')]

            # NEW: Extract link geometries and calculate bounding boxes
            self._extract_link_geometries(root)
            self._calculate_bounding_boxes()

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


    def _extract_link_geometries(self, root: ET.Element):
        """Extract geometry information from all links."""
        self.link_geometries = {}
        
        for link in root.findall('link'):
            link_name = link.get('name', 'unnamed')
            
            # Extract collision geometries (prioritize these for bounding boxes)
            collision_geoms = []
            for collision in link.findall('collision'):
                geom_data = self._parse_geometry_element(collision)
                if geom_data:
                    collision_geoms.append(geom_data)
            
            # Extract visual geometries as fallback
            visual_geoms = []
            for visual in link.findall('visual'):
                geom_data = self._parse_geometry_element(visual)
                if geom_data:
                    visual_geoms.append(geom_data)
            
            # Store geometry data
            self.link_geometries[link_name] = {
                'collision': collision_geoms,
                'visual': visual_geoms
            }

    def _parse_geometry_element(self, element: ET.Element) -> Optional[Dict]:
        """Parse a collision or visual element to extract geometry and transform."""
        geometry = element.find('geometry')
        if geometry is None:
            return None
            
        # Parse origin (transformation)
        origin = element.find('origin')
        transform = self._parse_origin(origin)
        
        # Parse geometry type
        geom_info = None
        
        # Box geometry
        box = geometry.find('box')
        if box is not None:
            size_str = box.get('size', '1 1 1')
            size = [float(x) for x in size_str.split()]
            geom_info = {
                'type': 'box',
                'size': size
            }
        
        # Cylinder geometry
        cylinder = geometry.find('cylinder')
        if cylinder is not None:
            radius = float(cylinder.get('radius', '1'))
            length = float(cylinder.get('length', '1'))
            geom_info = {
                'type': 'cylinder',
                'radius': radius,
                'length': length
            }
        
        # Sphere geometry
        sphere = geometry.find('sphere')
        if sphere is not None:
            radius = float(sphere.get('radius', '1'))
            geom_info = {
                'type': 'sphere',
                'radius': radius
            }
        
        # Mesh geometry
        mesh = geometry.find('mesh')
        if mesh is not None:
            filename = mesh.get('filename', '')
            scale_str = mesh.get('scale', '1 1 1')
            scale = [float(x) for x in scale_str.split()]
            geom_info = {
                'type': 'mesh',
                'filename': filename,
                'scale': scale
            }
        
        if geom_info:
            geom_info['transform'] = transform
            return geom_info
        
        return None

    def _parse_origin(self, origin: Optional[ET.Element]) -> Dict[str, List[float]]:
        """Parse origin element to get xyz translation and rpy rotation."""
        if origin is None:
            return {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]}
        
        xyz_str = origin.get('xyz', '0 0 0')
        rpy_str = origin.get('rpy', '0 0 0')
        
        xyz = [float(x) for x in xyz_str.split()]
        rpy = [float(x) for x in rpy_str.split()]
        
        return {'xyz': xyz, 'rpy': rpy}

    def _calculate_bounding_boxes(self):
        """Calculate bounding boxes for all links."""
        self.link_bounding_boxes = {}
        
        for link_name, geometries in self.link_geometries.items():
            # Prioritize collision geometries, fall back to visual
            geom_list = geometries['collision'] if geometries['collision'] else geometries['visual']
            
            if not geom_list:
                continue
                
            # Calculate bounding box for each geometry and combine
            all_bounds = []
            for geom in geom_list:
                bounds = self._calculate_geometry_bounds(geom)
                if bounds:
                    all_bounds.append(bounds)
            
            if all_bounds:
                combined_bounds = self._combine_bounds(all_bounds)
                self.link_bounding_boxes[link_name] = combined_bounds

    def _calculate_geometry_bounds(self, geom: Dict) -> Optional[Dict[str, List[float]]]:
        """Calculate bounding box for a single geometry."""
        geom_type = geom['type']
        transform = geom['transform']
        
        # Get local bounding box based on geometry type
        local_bounds = None
        
        if geom_type == 'box':
            size = geom['size']
            local_bounds = {
                'min': [-size[0]/2, -size[1]/2, -size[2]/2],
                'max': [size[0]/2, size[1]/2, size[2]/2]
            }
        
        elif geom_type == 'cylinder':
            radius = geom['radius']
            length = geom['length']
            local_bounds = {
                'min': [-radius, -radius, -length/2],
                'max': [radius, radius, length/2]
            }
        
        elif geom_type == 'sphere':
            radius = geom['radius']
            local_bounds = {
                'min': [-radius, -radius, -radius],
                'max': [radius, radius, radius]
            }
        
        elif geom_type == 'mesh':
            # For mesh, we can't calculate exact bounds without loading the file
            # Provide a reasonable default or indicate uncertainty
            scale = geom['scale']
            # Assume a unit cube and scale it
            local_bounds = {
                'min': [-scale[0]/2, -scale[1]/2, -scale[2]/2],
                'max': [scale[0]/2, scale[1]/2, scale[2]/2]
            }
            # Add metadata to indicate this is estimated
            local_bounds['estimated'] = True
            local_bounds['mesh_file'] = geom['filename']
        
        if local_bounds is None:
            return None
        
        # Apply transformation to bounds
        transformed_bounds = self._transform_bounds(local_bounds, transform)
        return transformed_bounds

    def _transform_bounds(self, bounds: Dict[str, List[float]], transform: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Apply transformation to bounding box."""
        xyz = transform['xyz']
        rpy = transform['rpy']
        
        # Get all 8 corners of the bounding box
        min_pt = bounds['min']
        max_pt = bounds['max']
        
        corners = [
            [min_pt[0], min_pt[1], min_pt[2]],
            [min_pt[0], min_pt[1], max_pt[2]],
            [min_pt[0], max_pt[1], min_pt[2]],
            [min_pt[0], max_pt[1], max_pt[2]],
            [max_pt[0], min_pt[1], min_pt[2]],
            [max_pt[0], min_pt[1], max_pt[2]],
            [max_pt[0], max_pt[1], min_pt[2]],
            [max_pt[0], max_pt[1], max_pt[2]]
        ]
        
        # Transform each corner
        transformed_corners = []
        for corner in corners:
            transformed_corner = self._transform_point(corner, xyz, rpy)
            transformed_corners.append(transformed_corner)
        
        # Find new min/max
        new_min = [min(corner[i] for corner in transformed_corners) for i in range(3)]
        new_max = [max(corner[i] for corner in transformed_corners) for i in range(3)]
        
        result = {
            'min': new_min,
            'max': new_max
        }
        
        # Preserve metadata
        if 'estimated' in bounds:
            result['estimated'] = bounds['estimated']
        if 'mesh_file' in bounds:
            result['mesh_file'] = bounds['mesh_file']
        
        return result

    def _transform_point(self, point: List[float], translation: List[float], rotation: List[float]) -> List[float]:
        """Transform a point by translation and rotation (RPY)."""
        # Apply rotation first (RPY = Roll, Pitch, Yaw)
        roll, pitch, yaw = rotation
        
        # Rotation matrices
        cos_r, sin_r = math.cos(roll), math.sin(roll)
        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        
        # Combined rotation matrix (ZYX order)
        r11 = cos_y * cos_p
        r12 = cos_y * sin_p * sin_r - sin_y * cos_r
        r13 = cos_y * sin_p * cos_r + sin_y * sin_r
        r21 = sin_y * cos_p
        r22 = sin_y * sin_p * sin_r + cos_y * cos_r
        r23 = sin_y * sin_p * cos_r - cos_y * sin_r
        r31 = -sin_p
        r32 = cos_p * sin_r
        r33 = cos_p * cos_r
        
        # Apply rotation
        x, y, z = point
        rotated_x = r11 * x + r12 * y + r13 * z
        rotated_y = r21 * x + r22 * y + r23 * z
        rotated_z = r31 * x + r32 * y + r33 * z
        
        # Apply translation
        final_x = rotated_x + translation[0]
        final_y = rotated_y + translation[1]
        final_z = rotated_z + translation[2]
        
        return [final_x, final_y, final_z]

    def _combine_bounds(self, bounds_list: List[Dict[str, List[float]]]) -> Dict[str, List[float]]:
        """Combine multiple bounding boxes into one."""
        if not bounds_list:
            return None
        
        all_mins = [bounds['min'] for bounds in bounds_list]
        all_maxs = [bounds['max'] for bounds in bounds_list]
        
        combined_min = [min(mins[i] for mins in all_mins) for i in range(3)]
        combined_max = [max(maxs[i] for maxs in all_maxs) for i in range(3)]
        
        result = {
            'min': combined_min,
            'max': combined_max
        }
        
        # Check if any bounds were estimated
        if any(bounds.get('estimated', False) for bounds in bounds_list):
            result['estimated'] = True
            result['mesh_files'] = [bounds['mesh_file'] for bounds in bounds_list 
                                  if bounds.get('mesh_file')]
        
        return result

    # ==================== DOF COUNTING METHODS ====================
    
    @staticmethod
    def get_joint_dof_count(joint_type: str) -> int:
        """Get the number of DOFs for a specific joint type.
        
        Args:
            joint_type: Type of joint ('revolute', 'prismatic', 'continuous', 'planar', 'floating', 'fixed')
            
        Returns:
            Number of DOFs for this joint type
        """
        dof_mapping = {
            'revolute': 1,
            'prismatic': 1, 
            'continuous': 1,
            'planar': 2,
            'floating': 6,
            'fixed': 0
        }
        return dof_mapping.get(joint_type.lower(), 0)
    
    def get_total_movable_dofs(self) -> int:
        """Get total number of DOFs for all movable joints.
        
        Returns:
            Total DOF count for movable joints
        """
        total_dofs = 0
        movable_joints = self.get_movable_joints()
        for joint in movable_joints:
            joint_type = joint['type']
            total_dofs += self.get_joint_dof_count(joint_type)
        return total_dofs
    
    def get_dof_counts_by_joint(self) -> Dict[str, int]:
        """Get DOF count for each joint by name.
        
        Returns:
            Dictionary mapping joint_name -> dof_count
        """
        dof_counts = {}
        for joint in self.joints:
            joint_name = joint['name']
            joint_type = joint['type']
            dof_counts[joint_name] = self.get_joint_dof_count(joint_type)
        return dof_counts
    
    def get_movable_joint_names_with_dofs(self) -> List[Dict[str, Union[str, int]]]:
        """Get movable joints with their DOF counts.
        
        Returns:
            List of dicts with 'name', 'type', and 'dofs' for each movable joint
        """
        movable_joints = self.get_movable_joints()
        result = []
        for joint in movable_joints:
            joint_info = joint.copy()
            joint_info['dofs'] = self.get_joint_dof_count(joint['type'])
            result.append(joint_info)
        return result

    # ==================== NEW PUBLIC METHODS ====================

    def get_link_bounding_boxes(self) -> Dict[str, Dict[str, List[float]]]:
        """Get bounding boxes for all links."""
        return self.link_bounding_boxes.copy()

    def get_link_bounding_box(self, link_name: str) -> Optional[Dict[str, List[float]]]:
        """Get bounding box for a specific link."""
        return self.link_bounding_boxes.get(link_name)

    def get_overall_bounding_box(self) -> Optional[Dict[str, List[float]]]:
        """Get overall bounding box encompassing all links."""
        if not self.link_bounding_boxes:
            return None
        
        all_bounds = list(self.link_bounding_boxes.values())
        return self._combine_bounds(all_bounds)

    def get_link_geometries(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Get raw geometry data for all links."""
        return self.link_geometries.copy()

    def print_bounding_box_summary(self, urdf_file: str):
        """Print a formatted summary of bounding boxes."""
        print(f"\n{'='*60}")
        print(f"BOUNDING BOX SUMMARY: {os.path.basename(urdf_file)}")
        print(f"{'='*60}")
        
        if not self.link_bounding_boxes:
            print("No bounding boxes calculated.")
            return
        
        print(f"\nLinks with bounding boxes: {len(self.link_bounding_boxes)}")
        
        # Overall bounding box
        overall = self.get_overall_bounding_box()
        if overall:
            print(f"\nOverall robot bounding box:")
            print(f"  Min: [{overall['min'][0]:.3f}, {overall['min'][1]:.3f}, {overall['min'][2]:.3f}]")
            print(f"  Max: [{overall['max'][0]:.3f}, {overall['max'][1]:.3f}, {overall['max'][2]:.3f}]")
            size = [overall['max'][i] - overall['min'][i] for i in range(3)]
            print(f"  Size: [{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]")
            if overall.get('estimated'):
                print(f"  Note: Contains estimated mesh bounds")
        
        # Individual link bounding boxes
        print(f"\nIndividual link bounding boxes:")
        for link_name, bounds in self.link_bounding_boxes.items():
            print(f"\n  {link_name}:")
            print(f"    Min: [{bounds['min'][0]:.3f}, {bounds['min'][1]:.3f}, {bounds['min'][2]:.3f}]")
            print(f"    Max: [{bounds['max'][0]:.3f}, {bounds['max'][1]:.3f}, {bounds['max'][2]:.3f}]")
            size = [bounds['max'][i] - bounds['min'][i] for i in range(3)]
            print(f"    Size: [{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]")
            if bounds.get('estimated'):
                print(f"    Note: Estimated from mesh file(s)")

    def print_full_summary(self, urdf_file: str):
        """Print both joint and bounding box summaries."""
        self.print_joint_summary(urdf_file)
        self.print_bounding_box_summary(urdf_file)


if __name__ == "__main__":
    """Example: Process a single URDF file with enhanced functionality."""
    extractor = URDFMetadataExtractor()

    # Replace with your actual URDF file path
    urdf_file = "/home/yuhao2024/Documents/Genesis_latest/genesis/assets/urdf/go2/urdf/go2.urdf"

    # Parse URDF (now extracts both joints and bounding boxes)
    joints = extractor.parse_urdf(urdf_file)

    # Original functionality still works
    joint_names = extractor.get_joint_names()
    print("Joint names in order:", joint_names)

    movable = extractor.get_movable_joints()
    print("\nMovable joints:")
    for joint in movable:
        print(f"  - {joint['name']} ({joint['type']})")
    
    # NEW: DOF counting functionality
    print("\n" + "="*60)
    print("NEW DOF COUNTING FUNCTIONALITY")
    print("="*60)
    
    # Test DOF counting methods
    total_dofs = extractor.get_total_movable_dofs()
    print(f"\nTotal movable DOFs: {total_dofs}")
    
    # Show DOF count per joint type
    joint_type_counts = {}
    dof_counts = extractor.get_dof_counts_by_joint()
    for joint_name, dof_count in dof_counts.items():
        joint_info = next((j for j in extractor.joints if j['name'] == joint_name), None)
        if joint_info:
            joint_type = joint_info['type']
            if joint_type not in joint_type_counts:
                joint_type_counts[joint_type] = {'count': 0, 'total_dofs': 0}
            joint_type_counts[joint_type]['count'] += 1
            joint_type_counts[joint_type]['total_dofs'] += dof_count
    
    print(f"\nJoint types and their DOF contributions:")
    for joint_type, data in joint_type_counts.items():
        dof_per_joint = extractor.get_joint_dof_count(joint_type)
        print(f"  - {joint_type}: {data['count']} joints × {dof_per_joint} DOF = {data['total_dofs']} DOFs")
    
    # Show movable joints with DOF info
    movable_with_dofs = extractor.get_movable_joint_names_with_dofs()
    print(f"\nMovable joints with DOF counts:")
    for joint in movable_with_dofs:
        print(f"  - {joint['name']} ({joint['type']}): {joint['dofs']} DOF(s)")

    # NEW: Bounding box functionality
    print("\n" + "="*60)
    print("NEW BOUNDING BOX FUNCTIONALITY")
    print("="*60)
    
    # Get bounding boxes
    bounding_boxes = extractor.get_link_bounding_boxes()
    print(f"\nFound bounding boxes for {len(bounding_boxes)} links")
    
    # Get overall bounding box
    overall_bbox = extractor.get_overall_bounding_box()
    if overall_bbox:
        print(f"\nOverall robot dimensions:")
        size = [overall_bbox['max'][i] - overall_bbox['min'][i] for i in range(3)]
        print(f"  Size: {size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f}")
    
    # Print detailed summaries
    extractor.print_full_summary(urdf_file)