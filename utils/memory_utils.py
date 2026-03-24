"""
显存优化工具模块
Memory Optimization Utilities

功能：
1. 显存监控
2. 梯度检查点
3. 批量大小自适应
4. 显存清理
"""

import torch
import gc
import psutil
import os


class MemoryMonitor:
    """显存监控器"""
    
    def __init__(self, device, logger=None):
        self.device = device
        self.logger = logger
        self.peak_memory = 0
    
    def _log(self, msg):
        if self.logger:
            self.logger.info("[MemoryMonitor] " + msg)
    
    def get_memory_info(self):
        """获取当前显存信息（GB）"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**3
            
            total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            free = total - allocated
            
            return {
                'allocated': allocated,
                'reserved': reserved,
                'max_allocated': max_allocated,
                'total': total,
                'free': free,
                'usage_percent': (allocated / total) * 100
            }
        else:
            return {'allocated': 0, 'reserved': 0, 'max_allocated': 0, 
                    'total': 0, 'free': 0, 'usage_percent': 0}
    
    def print_memory_summary(self, prefix=""):
        """打印显存摘要"""
        info = self.get_memory_info()
        self._log("{}显存状态: 已分配={:.2f}GB, 保留={:.2f}GB, 空闲={:.2f}GB, 总容量={:.2f}GB, 使用率={:.1f}%".format(
            prefix, info['allocated'], info['reserved'], info['free'], 
            info['total'], info['usage_percent']))
        
        self.peak_memory = max(self.peak_memory, info['allocated'])
    
    def reset_peak_memory(self):
        """重置峰值统计"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        self.peak_memory = 0
    
    def check_memory_leak(self, threshold_gb=1.0):
        """检查是否存在显存泄漏"""
        info = self.get_memory_info()
        leak = info['reserved'] - info['allocated']
        
        if leak > threshold_gb:
            self._log("⚠️ 可能存在显存泄漏: 保留但未分配 {:.2f}GB > 阈值 {:.2f}GB".format(
                leak, threshold_gb))
            return True
        return False


class MemoryOptimizer:
    """显存优化器"""
    
    def __init__(self, device, logger=None):
        self.device = device
        self.logger = logger
        self.monitor = MemoryMonitor(device, logger)
    
    def _log(self, msg):
        if self.logger:
            self.logger.info("[MemoryOptimizer] " + msg)
    
    def clear_cache(self):
        """清理显存缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            self._log("✓ 显存缓存已清理")
    
    def optimize_batch_size(self, initial_batch_size, memory_limit_gb=40.0):
        """
        根据显存限制自适应调整批量大小
        
        策略：
        - 如果显存使用率 > 90%，减半批量
        - 如果显存使用率 < 50%，可以尝试增加
        """
        info = self.monitor.get_memory_info()
        
        if info['usage_percent'] > 90:
            new_batch_size = max(1, initial_batch_size // 2)
            self._log("⚠️ 显存使用率过高 ({:.1f}%)，批量从 {} 降至 {}".format(
                info['usage_percent'], initial_batch_size, new_batch_size))
            return new_batch_size
        elif info['usage_percent'] < 50 and initial_batch_size < 64:
            new_batch_size = initial_batch_size * 2
            self._log("✓ 显存充足，批量从 {}增至 {}".format(
                initial_batch_size, new_batch_size))
            return new_batch_size
        else:
            return initial_batch_size
    
    def enable_gradient_checkpointing(self, model):
        """启用梯度检查点（以时间换空间）"""
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            self._log("✓ 梯度检查点已启用")
    
    def setup_memory_efficient_mode(self):
        """设置显存高效模式"""
        if torch.cuda.is_available():
            # 启用TF32（A100/H100上加速）
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # 启用CuDNN benchmark（固定输入尺寸时加速）
            torch.backends.cudnn.benchmark = True
            
            # 设置显存分配策略（减少碎片）
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
            self._log("✓ 显存高效模式已启用")
    
    def diagnose_oom(self, error_message):
        """
        诊断OOM错误原因
        
        输入:
            error_message: OOM错误信息
        
        返回:
            diagnosis: 诊断结果和建议
        """
        info = self.monitor.get_memory_info()
        
        diagnosis = []
        diagnosis.append("="*60)
        diagnosis.append("显存不足(OOM)诊断报告")
        diagnosis.append("="*60)
        diagnosis.append("错误信息: {}".format(error_message[:200]))
        diagnosis.append("")
        diagnosis.append("当前显存状态:")
        diagnosis.append("  - 已分配: {:.2f} GB".format(info['allocated']))
        diagnosis.append("  - 保留但未用: {:.2f} GB".format(info['reserved'] - info['allocated']))
        diagnosis.append("  - 总容量: {:.2f}GB".format(info['total']))
        diagnosis.append("  - 空闲: {:.2f} GB".format(info['free']))
        diagnosis.append("")
        
        # 分析原因
        diagnosis.append("可能原因:")
        if info['reserved'] - info['allocated'] > 5.0:
            diagnosis.append("  ✗ 显存碎片严重 ({:.2f}GB未释放)".format(
                info['reserved'] - info['allocated']))
            diagnosis.append("    建议: 设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        
        if "多进程" in error_message or "Process" in error_message:
            diagnosis.append("  ✗ 检测到多进程占用显存")
            diagnosis.append("    建议: 减少并行客户端数量或使用CPU offloading")
        
        if info['allocated'] > info['total'] * 0.95:
            diagnosis.append("  ✗ 模型+数据占用过大")
            diagnosis.append("    建议: 减小batch_size或使用梯度累积")
        
        diagnosis.append("")
        diagnosis.append("建议修复方案:")
        diagnosis.append("  1. 减小batch_size（当前建议: 1-4）")
        diagnosis.append("  2. 减少LLM层数（lm_layer_num=3或4）")
        diagnosis.append("  3. 启用梯度检查点")
        diagnosis.append("  4. 使用混合精度训练(FP16/BF16)")
        diagnosis.append("  5. 清理未使用的客户端进程")
        diagnosis.append("="*60)
        
        report = "\n".join(diagnosis)
        self._log(report)
        
        return report


def setup_memory_efficient_training(args, logger):
    """
    一键设置显存高效训练模式
    
    包含：
    1. 显存监控器
    2. 优化器配置
    3. 环境变量设置
    """
    optimizer = MemoryOptimizer(args.device, logger)
    
    # 启用显存高效模式
    optimizer.setup_memory_efficient_mode()
    
    # 初始显存检查
    optimizer.monitor.print_memory_summary("初始化前 ")
    
    return optimizer

