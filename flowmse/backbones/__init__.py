from .shared import BackboneRegistry

# 延迟导入：避免在只使用 DCUNet 时也要编译 NCSNpp 的 CUDA 算子
# 各骨干网络会在 BackboneRegistry.get_by_name() 时按需导入

def _register_backbones():
    """注册所有可用的骨干网络（延迟加载）"""
    # DCUNet 不需要自定义 CUDA 算子，优先注册
    try:
        from .dcunet import DCUNet
    except ImportError as e:
        import warnings
        warnings.warn(f"DCUNet 加载失败: {e}")
    
    # NCSNpp 需要 CUDA JIT 编译，可能失败
    try:
        from .ncsnpp import NCSNpp
    except (ImportError, OSError) as e:
        import warnings
        warnings.warn(f"NCSNpp 加载失败 (可能需要设置 CUDA_HOME): {e}")

_register_backbones()

__all__ = ['BackboneRegistry']
