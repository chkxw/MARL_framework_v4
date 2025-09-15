import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch

logger = logging.getLogger(__name__)


class ModelInference:
    """Ultra-efficient universal model inference with O(1) performance optimization"""

    def __init__(self, force_cpu: bool = False):
        """Initialize inference engine

        Args:
            force_cpu: Force CPU execution even if GPU available
        """
        self.force_cpu = force_cpu
        self.device = torch.device("cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Model state
        self.model = None
        self.model_type = None
        self.model_metadata = {}

        # ONNX specific
        self.onnx_session = None

        # PyTorch specific
        self.pytorch_model = None

        # Ultra-efficient decision system
        self.decision_functions = {}  # {input_type: lambda batch_size: optimal_config}
        self.crossover_thresholds = {}  # For reference and debugging
        self.is_performance_analyzed = False

        logger.info(f"ModelInference initialized - Device: {self.device}, Force CPU: {force_cpu}")

    def load_model(self, model_path: str) -> None:
        """Auto-detect and load ONNX or PyTorch model

        Args:
            model_path: Absolute path to model file (.onnx, .pt, .pth)
        """
        full_path = Path(model_path)
        if not full_path.exists():
            raise FileNotFoundError(f"Model not found: {full_path}")

        # Auto-detect model type
        self.model_type = self._detect_model_type(str(full_path))

        # Set current model path before loading (needed for metadata extraction)
        self._current_model_path = str(full_path)

        if self.model_type == "onnx":
            self._load_onnx_model(str(full_path))
        elif self.model_type == "pytorch":
            self._load_pytorch_model(str(full_path))
        else:
            raise ValueError(f"Unsupported model type for file: {model_path}")

        # Extract metadata
        self.model_metadata = self._extract_metadata()
        logger.info(f"Loaded {self.model_type} model: {model_path}")

    def predict(self, input_data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Ultra-fast universal prediction with O(1) optimization lookup

        Args:
            input_data: Input data (numpy array or torch tensor)

        Returns:
            Model output in same format as input
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        return self._run_inference(input_data)

    def _run_inference(self, input_data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Ultra-efficient inference with minimal overhead

        Uses O(1) lambda decision functions for optimal performance
        """
        # 1. Minimal detection overhead (O(1))
        input_type = self._detect_input_type(input_data)
        batch_size = input_data.shape[0] if len(input_data.shape) > 0 else 1

        # 2. O(1) decision using pre-computed lambda functions
        if self.is_performance_analyzed and input_type in self.decision_functions:
            optimal_config = self.decision_functions[input_type](batch_size)
        else:
            optimal_config = self._get_default_config(input_type)

        # 3. Execute with optimal configuration
        return self._execute_with_config(input_data, optimal_config)

    def _detect_input_type(self, input_data: Union[np.ndarray, torch.Tensor]) -> str:
        """Minimal overhead input type detection"""
        if isinstance(input_data, np.ndarray):
            return "numpy"
        elif hasattr(input_data, 'device'):
            device_str = str(input_data.device)
            return "tensor_gpu" if "cuda" in device_str else "tensor_cpu"
        else:
            return "tensor_cpu"

    def _get_default_config(self, input_type: str) -> str:
        """Get default configuration when no performance analysis available"""
        if self.force_cpu:
            return "cpu"
        elif input_type == "tensor_gpu":
            return "gpu"  # Keep GPU tensors on GPU
        elif not self.force_cpu and torch.cuda.is_available():
            # For GPU instances, use GPU execution (handles data transfer automatically)
            return "gpu"
        else:
            return "cpu"  # Default to CPU for numpy and CPU tensors

    def _execute_with_config(
        self, input_data: Union[np.ndarray, torch.Tensor], config: str
    ) -> Union[np.ndarray, torch.Tensor]:
        """Execute inference with optimal configuration"""
        if config == "gpu" and not self.force_cpu and torch.cuda.is_available():
            return self._execute_gpu_inference(input_data)
        else:
            return self._execute_cpu_inference(input_data)

    def _execute_gpu_inference(self, input_data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Execute inference optimized for GPU"""
        if self.model_type == "onnx":
            # Ensure ONNX session has GPU providers
            providers = self.onnx_session.get_providers()
            if 'CUDAExecutionProvider' not in providers:
                self._ensure_gpu_onnx_session()

            # Convert to numpy for ONNX (ONNX Runtime handles GPU internally)
            if isinstance(input_data, torch.Tensor):
                original_device = input_data.device
                input_np = input_data.cpu().numpy()
            else:
                original_device = None
                input_np = input_data

            # Run inference
            input_name = self.onnx_session.get_inputs()[0].name
            outputs = self.onnx_session.run(None, {input_name: input_np})

            # Convert back to original format
            if original_device is not None:
                return torch.from_numpy(outputs[0]).to(original_device)
            else:
                return outputs[0]

        elif self.model_type == "pytorch":
            # Convert to GPU tensor if needed
            if isinstance(input_data, np.ndarray):
                input_tensor = torch.from_numpy(input_data).to(self.device)
            else:
                input_tensor = input_data.to(self.device)

            return self._run_pytorch_inference(input_tensor)

    def _execute_cpu_inference(self, input_data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Execute inference optimized for CPU"""
        if self.model_type == "onnx":
            # Convert to numpy for ONNX
            if isinstance(input_data, torch.Tensor):
                original_device = input_data.device
                input_np = input_data.cpu().numpy()
            else:
                original_device = None
                input_np = input_data

            # Run inference
            input_name = self.onnx_session.get_inputs()[0].name
            outputs = self.onnx_session.run(None, {input_name: input_np})

            # Convert back to original format
            if original_device is not None:
                return torch.from_numpy(outputs[0]).to(original_device)
            else:
                return outputs[0]

        elif self.model_type == "pytorch":
            # Use CPU device - ensure both model and input are on CPU
            if isinstance(input_data, np.ndarray):
                input_tensor = torch.from_numpy(input_data)
            else:
                input_tensor = input_data.cpu()

            # Temporarily move model to CPU for CPU inference
            original_device = (
                next(self.pytorch_model.parameters()).device if hasattr(self.pytorch_model, 'parameters') else None
            )
            if original_device is not None and 'cuda' in str(original_device):
                self.pytorch_model = self.pytorch_model.cpu()
                output = self._run_pytorch_inference(input_tensor)
                # Move model back to original device
                self.pytorch_model = self.pytorch_model.to(original_device)
                return output
            else:
                return self._run_pytorch_inference(input_tensor)

    def _run_pytorch_inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run PyTorch inference"""
        with torch.no_grad():
            if isinstance(self.pytorch_model, torch.nn.Module):
                output = self.pytorch_model(input_tensor)
            elif isinstance(self.pytorch_model, torch.Tensor):
                pytorch_tensor = self.pytorch_model.to(input_tensor.device)
                if pytorch_tensor.dim() == 1:
                    output = input_tensor * pytorch_tensor
                else:
                    output = torch.matmul(input_tensor, pytorch_tensor)
            elif isinstance(self.pytorch_model, dict):
                # Handle state dict by performing sequential operations
                output = input_tensor

                # Sort keys to ensure consistent order
                sorted_keys = sorted(
                    [k for k in self.pytorch_model.keys() if not k.startswith('ln') and not k.startswith('bias')]
                )

                for key in sorted_keys:
                    weight = self.pytorch_model[key].to(input_tensor.device)

                    if weight.dim() == 1:
                        # 1D tensor - element-wise multiplication or bias
                        if output.shape[-1] == weight.shape[0]:
                            output = output * weight
                    elif weight.dim() == 2:
                        # 2D tensor - matrix multiplication
                        if output.shape[-1] == weight.shape[0]:
                            output = torch.matmul(output, weight)
                        elif output.shape[-1] == weight.shape[1]:
                            output = torch.matmul(output, weight.T)

                    # Apply activation function every few layers to simulate real networks
                    if 'layer' in key or 'ffn' in key or 'attention' in key:
                        output = torch.relu(output)
            else:
                raise RuntimeError(f"Unsupported PyTorch model type: {type(self.pytorch_model)}")

        return output

    def _ensure_gpu_onnx_session(self) -> None:
        """Ensure ONNX session has GPU providers"""
        if self.model_type == "onnx" and torch.cuda.is_available() and not self.force_cpu:
            current_model_path = Path(self._current_model_path)
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.onnx_session = ort.InferenceSession(str(current_model_path), providers=providers)

    def record_performance_analysis(
        self, sample_input: Union[np.ndarray, torch.Tensor], max_batch_size: int = 128, verbose: bool = False
    ) -> None:
        """Record performance analysis and build ultra-fast lambda decision functions

        Args:
            sample_input: Sample input for analysis
            max_batch_size: Maximum batch size to test
            verbose: Print analysis progress
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        if verbose:
            print(f"📊 Recording performance analysis for ultra-efficient inference")

        # Run performance analysis
        current_model_full_path = Path(self._current_model_path)
        results = self.__class__.analyze_performance(
            str(current_model_full_path), sample_input, max_batch_size, verbose=verbose
        )

        # Build ultra-efficient lambda decision functions
        self._build_decision_functions(results)
        self.is_performance_analyzed = True

        if verbose:
            print(f"✅ Ultra-efficient decision functions built")
            print(f"   Input types optimized: {list(self.decision_functions.keys())}")
            print(f"   Crossover thresholds: {self.crossover_thresholds}")

    def _build_decision_functions(self, analysis_results: Dict[str, Any]) -> None:
        """Build ultra-fast lambda decision functions from analysis results with infinity threshold handling"""
        self.decision_functions = {}
        self.crossover_thresholds = {}

        if "configurations" not in analysis_results:
            return

        cpu_config = analysis_results["configurations"].get("cpu", {})
        gpu_config = analysis_results["configurations"].get("gpu", {})

        if not cpu_config.get("timings") or not gpu_config.get("timings"):
            return

        cpu_timings = {t["batch_size"]: t["per_sample_ms"] for t in cpu_config["timings"]}
        gpu_timings = {t["batch_size"]: t["per_sample_ms"] for t in gpu_config["timings"]}
        max_batch_tested = max(cpu_timings.keys()) if cpu_timings else 128

        # Build decision functions for each input type
        input_types = ["numpy", "tensor_cpu", "tensor_gpu"]

        for input_type in input_types:
            if input_type == "tensor_gpu":
                # GPU tensors should always stay on GPU to avoid transfers
                self.decision_functions[input_type] = lambda bs: "gpu"
                self.crossover_thresholds[input_type] = 1
            else:
                crossover_batch = self._find_crossover_point(cpu_timings, gpu_timings, max_batch_tested)

                if crossover_batch is None:
                    # GPU never beneficial - set to infinity
                    self.decision_functions[input_type] = lambda bs: "cpu"
                    self.crossover_thresholds[input_type] = float('inf')
                elif crossover_batch > max_batch_tested * 2:
                    # Crossover beyond practical range - treat as never beneficial
                    self.decision_functions[input_type] = lambda bs: "cpu"
                    self.crossover_thresholds[input_type] = float('inf')
                else:
                    # Normal crossover case - create lambda with captured threshold
                    self.decision_functions[input_type] = lambda bs, threshold=crossover_batch: (
                        "gpu" if bs >= threshold else "cpu"
                    )
                    self.crossover_thresholds[input_type] = crossover_batch

    def _find_crossover_point(
        self, cpu_timings: Dict[int, float], gpu_timings: Dict[int, float], max_batch_tested: int
    ) -> Optional[int]:
        """Find batch size where GPU becomes faster than CPU with enhanced trend analysis"""
        common_batch_sizes = sorted(set(cpu_timings.keys()) & set(gpu_timings.keys()))

        # Find actual crossover in tested range
        for batch_size in common_batch_sizes:
            if gpu_timings[batch_size] < cpu_timings[batch_size]:
                return batch_size

        # No crossover found in tested range - check if trend suggests future crossover
        if self._extrapolate_crossover_trend(cpu_timings, gpu_timings, max_batch_tested):
            # Trend suggests crossover beyond tested range - return conservative estimate
            return max_batch_tested * 2

        # No crossover likely - return None for "never beneficial"
        return None

    def _extrapolate_crossover_trend(
        self, cpu_timings: Dict[int, float], gpu_timings: Dict[int, float], max_batch_tested: int
    ) -> bool:
        """Analyze if trend suggests future GPU benefit beyond tested range"""
        batch_sizes = sorted(set(cpu_timings.keys()) & set(gpu_timings.keys()))

        if len(batch_sizes) < 3:
            return False

        # Calculate speedup trend over last few batch sizes
        recent_batches = batch_sizes[-3:]  # Last 3 points
        speedup_trend = []

        for batch in recent_batches:
            if gpu_timings[batch] > 0:  # Avoid division by zero
                speedup = cpu_timings[batch] / gpu_timings[batch]
                speedup_trend.append(speedup)

        if len(speedup_trend) < 2:
            return False

        # Check if speedup is increasing (approaching 1.0)
        trend_slope = (speedup_trend[-1] - speedup_trend[0]) / len(speedup_trend)

        # If trend is positive and approaching 1.0, extrapolate potential crossover
        is_improving = trend_slope > 0.01  # Positive trend
        approaching_crossover = speedup_trend[-1] > 0.7  # Getting close to 1.0

        return is_improving and approaching_crossover

    def get_performance_status(self) -> Dict[str, Any]:
        """Get current performance analysis status with enhanced infinity threshold handling"""
        status = {
            "is_analyzed": self.is_performance_analyzed,
            "decision_functions_count": len(self.decision_functions),
            "input_types_optimized": list(self.decision_functions.keys()),
            "crossover_thresholds": {},
            "gpu_beneficial_types": [],
            "cpu_only_types": [],
            "model_type": self.model_type,
        }

        # Process thresholds with infinity handling
        for input_type, threshold in self.crossover_thresholds.items():
            if threshold == float('inf'):
                status["crossover_thresholds"][input_type] = "never"
                status["cpu_only_types"].append(input_type)
            else:
                status["crossover_thresholds"][input_type] = threshold
                status["gpu_beneficial_types"].append(input_type)

        return status

    def predict_with_explanation(self, input_data: Union[np.ndarray, torch.Tensor], verbose: bool = False) -> tuple:
        """Predict with explanation of ultra-fast decision process"""
        # Analyze input (same minimal overhead as inference)
        input_type = self._detect_input_type(input_data)
        batch_size = input_data.shape[0] if len(input_data.shape) > 0 else 1

        # Get decision
        if self.is_performance_analyzed and input_type in self.decision_functions:
            optimal_config = self.decision_functions[input_type](batch_size)
            decision_method = "lambda_function"
            threshold = self.crossover_thresholds.get(input_type)
        else:
            optimal_config = self._get_default_config(input_type)
            decision_method = "default_fallback"
            threshold = None

        explanation = {
            "input_type": input_type,
            "batch_size": batch_size,
            "optimal_config": optimal_config,
            "decision_method": decision_method,
            "crossover_threshold": threshold,
            "ultra_efficient": self.is_performance_analyzed,
        }

        if verbose:
            print(f"⚡ Ultra-Efficient Inference Decision:")
            print(f"   Input: {input_type}, batch_size={batch_size}")
            print(f"   Decision: {optimal_config} (via {decision_method})")

            if threshold == float('inf'):
                print(f"   Analysis: GPU never beneficial for {input_type} (within tested range)")
            elif threshold is not None:
                print(f"   Threshold: GPU faster when batch_size >= {threshold}")

        result = self.predict(input_data)
        return result, explanation

    # Keep existing utility methods
    def discover_models(self, model_dir: str = "exported_models") -> Dict[str, List[str]]:
        """Discover available model files in model directory

        Args:
            model_dir: Directory path to search for model files

        Returns:
            Dictionary with lists of ONNX and PyTorch model file paths
        """
        models = {"onnx": [], "pytorch": []}
        model_path = Path(model_dir)

        if not model_path.exists():
            logger.warning(f"Model directory not found: {model_path}")
            return models

        for file_path in model_path.iterdir():
            if file_path.is_file():
                if file_path.suffix.lower() == '.onnx':
                    models["onnx"].append(str(file_path))
                elif file_path.suffix.lower() in ['.pt', '.pth']:
                    models["pytorch"].append(str(file_path))

        return models

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model metadata"""
        if self.model is None:
            return {"error": "No model loaded"}

        # Base info common to all model types
        info = {
            "type": self.model_type,
            "device": str(self.device),
            "metadata": self.model_metadata,
            "ultra_efficient": self.is_performance_analyzed,
            "decision_functions": len(self.decision_functions),
        }

        # Add model-specific shape information
        if self.model_type == "onnx" and self.model_metadata.get('inputs'):
            # ONNX models: extract shape from graph metadata
            info["input_shape"] = self.model_metadata['inputs'][0].get('shape', [])
            info["output_shape"] = (
                self.model_metadata['outputs'][0].get('shape', []) if self.model_metadata.get('outputs') else []
            )

        elif self.model_type == "pytorch":
            # PyTorch models: include detailed metadata and verified shape info
            info["pytorch_info"] = self.model_metadata

            # Add shape information if input shape is verified
            if self.model_metadata.get('input_shape_verified'):
                verified_input_size = self.model_metadata['verified_input_size']
                sample_output_shape = self.model_metadata['sample_output_shape']
                supports_dynamic_batch = self.model_metadata.get('supports_dynamic_batch', False)

                # Set shape format based on dynamic batch support
                if supports_dynamic_batch:
                    # Use dynamic batch format for both input and output
                    info["input_shape"] = ['batch_size', verified_input_size]
                    info["output_shape"] = ['batch_size'] + sample_output_shape[1:]  # Keep all dims except batch
                else:
                    # Use concrete shapes from sample run
                    info["input_shape"] = [1, verified_input_size]
                    info["output_shape"] = sample_output_shape

        return info

    @classmethod
    def analyze_performance(
        cls,
        model_path: str,
        sample_input: Union[np.ndarray, torch.Tensor],
        max_batch_size: int = 128,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Analyze model performance across batch sizes and devices (same as before)"""
        model_path = Path(model_path)

        if verbose:
            print(f"🔍 Analyzing performance for: {model_path.name}")
            print(f"   Sample input shape: {sample_input.shape}")
            print(f"   Max batch size: {max_batch_size}")

        results = {
            "model_path": str(model_path),
            "sample_input_info": {
                "shape": tuple(sample_input.shape),
                "type": type(sample_input).__name__,
                "device": str(sample_input.device) if hasattr(sample_input, 'device') else 'cpu',
            },
            "configurations": {},
        }

        # Test CPU configuration
        if verbose:
            print("\n📊 Testing CPU configuration...")
        try:
            cpu_inference = cls(force_cpu=True)
            cpu_inference.load_model(str(model_path))
            batch_sizes = cpu_inference._generate_batch_sizes(max_batch_size)
            cpu_results = cpu_inference._benchmark_batch_sizes(batch_sizes, sample_input, verbose=verbose)
            results["configurations"]["cpu"] = cpu_results
        except Exception as e:
            if verbose:
                print(f"   ❌ CPU test failed: {e}")

        # Test GPU configuration (if available)
        if torch.cuda.is_available():
            if verbose:
                print("\n🚀 Testing GPU configuration...")
            try:
                gpu_inference = cls(force_cpu=False)
                gpu_inference.load_model(str(model_path))
                batch_sizes = gpu_inference._generate_batch_sizes(max_batch_size)
                gpu_results = gpu_inference._benchmark_batch_sizes(batch_sizes, sample_input, verbose=verbose)
                results["configurations"]["gpu"] = gpu_results
            except Exception as e:
                if verbose:
                    print(f"   ❌ GPU test failed: {e}")

        # Analyze results
        results["optimal_thresholds"] = cls._find_optimal_thresholds(results)
        results["recommendations"] = cls._generate_recommendations(results, sample_input)

        if verbose:
            print(f"\n💡 Recommendations:")
            for key, rec in results["recommendations"].items():
                print(f"   {key}: {rec}")

        return results

    # Private helper methods (keeping essential ones from original)
    def _detect_model_type(self, model_path: str) -> str:
        """Detect if model is ONNX or PyTorch"""
        suffix = Path(model_path).suffix.lower()
        if suffix == '.onnx':
            return "onnx"
        elif suffix in ['.pt', '.pth']:
            return "pytorch"
        else:
            raise ValueError(f"Unsupported file extension: {suffix}")

    def _load_onnx_model(self, model_path: str) -> None:
        """Load ONNX model internally"""
        full_path = Path(model_path)

        if self.force_cpu:
            providers = ['CPUExecutionProvider']
        else:
            providers = (
                ['CUDAExecutionProvider', 'CPUExecutionProvider']
                if torch.cuda.is_available()
                else ['CPUExecutionProvider']
            )

        self.onnx_session = ort.InferenceSession(str(full_path), providers=providers)
        self.model = self.onnx_session

    def _load_pytorch_model(self, model_path: str) -> None:
        """Load PyTorch model internally"""
        full_path = Path(model_path)

        try:
            # Try loading with weights_only=False for nn.Module objects
            self.pytorch_model = torch.load(full_path, map_location=self.device, weights_only=False)
        except Exception:
            # Fallback to default loading
            self.pytorch_model = torch.load(full_path, map_location=self.device)

        if hasattr(self.pytorch_model, 'eval'):
            self.pytorch_model.eval()

        self.model = self.pytorch_model

    def _extract_metadata(self) -> Dict[str, Any]:
        """Extract metadata from loaded model"""
        if self.model_type == "onnx":
            return self._extract_onnx_metadata()
        elif self.model_type == "pytorch":
            return self._extract_pytorch_metadata()
        else:
            return {}

    def _extract_onnx_metadata(self) -> Dict[str, Any]:
        """Extract ONNX metadata"""
        try:
            # Validate model path
            if not hasattr(self, '_current_model_path') or not self._current_model_path:
                logger.warning("No current model path set for ONNX metadata extraction")
                return {}

            current_model_path = Path(self._current_model_path)
            if not current_model_path.exists():
                logger.warning(f"ONNX model path does not exist: {current_model_path}")
                return {}

            if current_model_path.suffix.lower() != '.onnx':
                logger.warning(f"Not an ONNX model: {current_model_path}")
                return {}

            # Load model and initialize metadata
            model = onnx.load(str(current_model_path))
            metadata = {
                'inputs': [],
                'outputs': [],
                'producer_name': model.producer_name,
                'producer_version': model.producer_version,
            }

            # Helper function to parse tensor info (reduces code duplication)
            def parse_tensor_info(tensor_info):
                tensor_meta = {'name': tensor_info.name, 'type': tensor_info.type.tensor_type.elem_type, 'shape': []}
                for dim in tensor_info.type.tensor_type.shape.dim:
                    if dim.dim_value:
                        tensor_meta['shape'].append(dim.dim_value)
                    elif dim.dim_param:
                        tensor_meta['shape'].append(dim.dim_param)
                    else:
                        tensor_meta['shape'].append(-1)
                return tensor_meta

            # Parse inputs and outputs
            metadata['inputs'] = [parse_tensor_info(inp) for inp in model.graph.input]
            metadata['outputs'] = [parse_tensor_info(out) for out in model.graph.output]

            return metadata

        except Exception as e:
            logger.warning(f"Failed to extract ONNX metadata: {e}")
            logger.warning(f"Current model path: {getattr(self, '_current_model_path', 'Not set')}")
            return {}

    def _extract_pytorch_metadata(self) -> Dict[str, Any]:
        """Extract PyTorch metadata with enhanced input shape detection"""
        metadata = {'type': str(type(self.pytorch_model)), 'device': str(self.device)}

        if isinstance(self.pytorch_model, torch.nn.Module):
            metadata['model_type'] = 'nn.Module'
            metadata['parameters'] = sum(p.numel() for p in self.pytorch_model.parameters())

            # Try multiple detection methods
            self._detect_nn_module_input_shape(metadata)
            self._verify_detected_input_shape(metadata)

        elif isinstance(self.pytorch_model, torch.Tensor):
            metadata['model_type'] = 'Tensor'
            metadata['shape'] = list(self.pytorch_model.shape)
            metadata['dtype'] = str(self.pytorch_model.dtype)

        elif isinstance(self.pytorch_model, dict):
            metadata['model_type'] = 'state_dict'
            metadata['keys'] = list(self.pytorch_model.keys())
            self._detect_state_dict_input_shape(metadata)

        else:
            metadata['model_type'] = 'unknown'

        return metadata

    def _detect_nn_module_input_shape(self, metadata: Dict[str, Any]) -> None:
        """Detect input shape for nn.Module using multiple methods"""
        # Method 1: First parameter analysis (for FC layers)
        if not metadata.get('input_detection_method'):
            try:
                first_parameter = next(self.pytorch_model.parameters())
                if len(first_parameter.shape) >= 2:
                    metadata['potential_input_size'] = first_parameter.shape[1]
                    metadata['input_detection_method'] = 'first_parameter'
            except (StopIteration, IndexError):
                pass

        # Method 2: TorchScript graph introspection (highest priority)
        if not metadata.get('torchscript_input_shape'):
            try:
                if hasattr(self.pytorch_model, 'graph'):
                    graph = self.pytorch_model.graph
                    inputs = list(graph.inputs())
                    if len(inputs) > 1:  # Skip 'self' parameter
                        input_node = inputs[1]
                        if hasattr(input_node, 'type') and hasattr(input_node.type(), 'sizes'):
                            input_shape = list(input_node.type().sizes())
                            metadata['torchscript_input_shape'] = input_shape
                            metadata['input_detection_method'] = 'torchscript_graph'
            except:
                pass

        # Method 3: Model string parsing (fallback)
        if not metadata.get('suggested_input_size'):
            try:
                model_str = str(self.pytorch_model)
                if 'Linear(' in model_str:
                    import re

                    linear_matches = re.findall(r'Linear\((?:in_features=)?(\d+)', model_str)
                    if linear_matches:
                        metadata['linear_input_sizes'] = [int(m) for m in linear_matches]
                        metadata['suggested_input_size'] = int(linear_matches[0])
                        if not metadata.get('input_detection_method'):
                            metadata['input_detection_method'] = 'model_string_parsing'
            except:
                pass

    def _detect_state_dict_input_shape(self, metadata: Dict[str, Any]) -> None:
        """Detect input shape from state_dict"""
        try:
            # Look for first layer weights (common patterns)
            first_layer_patterns = ['0.weight', 'first', 'input', 'fc1.weight', 'linear.weight']
            first_layer_keys = [
                k for k in self.pytorch_model.keys() if any(pattern in k for pattern in first_layer_patterns)
            ]

            if first_layer_keys:
                first_weight = self.pytorch_model[first_layer_keys[0]]
                if len(first_weight.shape) >= 2:
                    metadata['potential_input_size'] = first_weight.shape[1]
                    metadata['input_detection_method'] = 'state_dict_first_weight'
                    metadata['first_layer_key'] = first_layer_keys[0]
                    metadata['input_shape_verified'] = False
                    metadata['verification_note'] = 'Cannot verify state_dict input shape without model structure'
        except:
            pass

    def _verify_detected_input_shape(self, metadata: Dict[str, Any]) -> None:
        """Verify detected input shape with sample runs and check dynamic batch support"""
        # Determine best detected input size (prioritize TorchScript > string parsing > first parameter)
        detected_input_size = None
        detection_method = None

        if 'torchscript_input_shape' in metadata and len(metadata['torchscript_input_shape']) >= 2:
            detected_input_size = metadata['torchscript_input_shape'][1]
            detection_method = 'torchscript_graph'
        elif 'suggested_input_size' in metadata:
            detected_input_size = metadata['suggested_input_size']
            detection_method = 'model_string_parsing'
        elif 'potential_input_size' in metadata:
            detected_input_size = metadata['potential_input_size']
            detection_method = 'first_parameter'

        if detected_input_size is not None:
            try:
                with torch.no_grad():
                    # Test batch size 1 first
                    test_input_1 = torch.randn(1, detected_input_size, device=self.device)
                    test_output_1 = self.pytorch_model(test_input_1)

                    # Mark as verified for batch size 1
                    metadata['verified_input_size'] = detected_input_size
                    metadata['verified_method'] = detection_method
                    metadata['sample_output_shape'] = list(test_output_1.shape)
                    metadata['input_shape_verified'] = True

                    # Test batch size 2 to check dynamic batch support
                    try:
                        test_input_2 = torch.randn(2, detected_input_size, device=self.device)
                        test_output_2 = self.pytorch_model(test_input_2)

                        # Check if output batch dimension scales correctly
                        if test_output_2.shape[0] == 2 and test_output_2.shape[1:] == test_output_1.shape[1:]:
                            metadata['supports_dynamic_batch'] = True
                        else:
                            metadata['supports_dynamic_batch'] = False

                    except Exception:
                        # If batch size 2 fails, model doesn't support dynamic batching
                        metadata['supports_dynamic_batch'] = False

            except Exception as verification_error:
                # Mark as unverified
                metadata['input_shape_verified'] = False
                metadata['verification_error'] = str(verification_error)
                metadata['failed_input_size'] = detected_input_size

    def _benchmark_batch_sizes(
        self,
        batch_sizes: List[int],
        sample_input: Union[np.ndarray, torch.Tensor],
        warmup_runs: int = 5,
        timing_runs: int = 20,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Benchmark performance across batch sizes"""
        results = {"batch_sizes": batch_sizes, "timings": [], "device": str(self.device), "force_cpu": self.force_cpu}

        if len(sample_input.shape) > 1:
            input_shape = sample_input.shape[1:]
        else:
            input_shape = sample_input.shape

        for batch_size in batch_sizes:
            try:
                if isinstance(sample_input, np.ndarray):
                    test_data = np.random.randn(batch_size, *input_shape).astype(np.float32)
                else:
                    test_data = torch.randn(batch_size, *input_shape, dtype=torch.float32, device=sample_input.device)

                # Warmup
                for _ in range(warmup_runs):
                    self._run_inference(test_data)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # Timing
                start_time = time.perf_counter()
                for _ in range(timing_runs):
                    self._run_inference(test_data)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                end_time = time.perf_counter()

                avg_time_ms = (end_time - start_time) * 1000 / timing_runs
                per_sample_ms = avg_time_ms / batch_size

                timing_result = {
                    'batch_size': batch_size,
                    'total_time_ms': avg_time_ms,
                    'per_sample_ms': per_sample_ms,
                    'throughput_samples_per_sec': 1000 / per_sample_ms,
                }

                results["timings"].append(timing_result)

                if verbose and batch_size in [1, 8, 32, 128]:
                    print(f"   Batch {batch_size:3d}: {avg_time_ms:6.3f}ms total, {per_sample_ms:6.3f}ms/sample")

            except Exception as e:
                if verbose:
                    print(f"   Batch {batch_size:3d}: Error - {e}")
                continue

        return results

    def _generate_batch_sizes(self, max_batch_size: int) -> List[int]:
        """Generate dynamic batch sizes for testing"""
        if max_batch_size <= 20:
            return list(range(1, max_batch_size + 1))
        else:
            log_points = np.logspace(np.log10(2), np.log10(max_batch_size), 19)
            batch_sizes = [1] + [int(round(x)) for x in log_points]
            batch_sizes = sorted(list(set(batch_sizes)))
            if batch_sizes[-1] != max_batch_size:
                batch_sizes.append(max_batch_size)
            return batch_sizes

    @staticmethod
    def _find_optimal_thresholds(results: Dict[str, Any]) -> Dict[str, Any]:
        """Find optimal device switching thresholds"""
        thresholds = {}

        if "cpu" in results["configurations"] and "gpu" in results["configurations"]:
            cpu_timings = results["configurations"]["cpu"]["timings"]
            gpu_timings = results["configurations"]["gpu"]["timings"]

            crossover_batch = None
            for cpu_timing, gpu_timing in zip(cpu_timings, gpu_timings):
                if (
                    cpu_timing['batch_size'] == gpu_timing['batch_size']
                    and gpu_timing['per_sample_ms'] < cpu_timing['per_sample_ms']
                ):
                    crossover_batch = cpu_timing['batch_size']
                    break

            thresholds["gpu_crossover_batch"] = crossover_batch
            if crossover_batch:
                thresholds["recommendation"] = (
                    f"Use CPU for batch < {crossover_batch}, GPU for batch >= {crossover_batch}"
                )
            else:
                thresholds["recommendation"] = "CPU always faster for this model/input combination"

        return thresholds

    @staticmethod
    def _generate_recommendations(
        results: Dict[str, Any], sample_input: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, str]:
        """Generate practical usage recommendations"""
        recommendations = {}

        input_info = results["sample_input_info"]
        if input_info["type"] == "ndarray":
            recommendations["input_based"] = "Use CPU (numpy arrays work well with CPU)"
        elif "cuda" in input_info["device"]:
            recommendations["input_based"] = "Use GPU (GPU tensors avoid transfers)"
        else:
            recommendations["input_based"] = "Use CPU (CPU tensors work well with CPU)"

        if "optimal_thresholds" in results and results["optimal_thresholds"].get("gpu_crossover_batch"):
            crossover = results["optimal_thresholds"]["gpu_crossover_batch"]
            recommendations["performance_based"] = f"GPU becomes faster at batch size {crossover}+"
        else:
            recommendations["performance_based"] = "CPU preferred for all tested batch sizes"

        sample_batch = input_info["shape"][0] if input_info["shape"] else 1
        if (
            results.get("optimal_thresholds", {}).get("gpu_crossover_batch")
            and sample_batch >= results["optimal_thresholds"]["gpu_crossover_batch"]
        ):
            recommendations["overall"] = "Use GPU configuration (force_cpu=False)"
        else:
            recommendations["overall"] = "Use CPU configuration (force_cpu=True)"

        return recommendations


def test_model_inference():
    """Test ultra-efficient model inference"""
    print("=== Testing Ultra-Efficient Model Inference ===")

    inference = ModelInference()
    models = inference.discover_models("exported_models")
    print(f"Discovered models: {models}")

    # Test ONNX model first (has reliable metadata extraction)
    if models["onnx"]:
        print(f"\n--- Testing ONNX Model: {models['onnx'][0]} ---")
        try:
            inference.load_model(models["onnx"][0])
            model_info = inference.get_model_info()
            print(f"Model info: {model_info}")

            # Extract input shape from ONNX metadata
            if 'inputs' in model_info['metadata'] and model_info['metadata']['inputs']:
                input_shape = model_info['metadata']['inputs'][0]['shape']
                print(f"Expected input shape from ONNX metadata: {input_shape}")

                # Use the shape info to create proper input
                if len(input_shape) >= 2 and input_shape[1] != -1:
                    dummy_input = torch.randn(1, input_shape[1])
                    output = inference.predict(dummy_input)
                    print(f"✅ Success with input shape: {dummy_input.shape}, Output shape: {output.shape}")
                else:
                    print("❌ Cannot determine input size from ONNX metadata")
            else:
                print("❌ No input metadata found in ONNX model")

        except Exception as e:
            print(f"Error testing ONNX model: {e}")

    # Test PyTorch model (metadata extraction is less reliable)
    if models["pytorch"]:
        print(f"\n--- Testing PyTorch Model: {models['pytorch'][0]} ---")
        try:
            inference.load_model(models["pytorch"][0])
            model_info = inference.get_model_info()
            print(f"Model info: {model_info}")

            # PyTorch models don't have reliable input shape extraction
            # Try common sizes based on the error message
            expected_input_size = 45
            dummy_input = torch.randn(1, expected_input_size)
            output = inference.predict(dummy_input)
            print(f"Input shape: {dummy_input.shape}, Output shape: {output.shape}")

        except Exception as e:
            print(f"Error testing PyTorch model: {e}")


if __name__ == "__main__":
    test_model_inference()
