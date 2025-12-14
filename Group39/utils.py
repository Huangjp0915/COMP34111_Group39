"""
工具函数模块
功能：
1. 配置日志系统 (文件输出 DEBUG 信息，控制台输出 INFO 信息)。
2. 提供棋盘哈希和辅助计算功能。
"""
import logging
import os
import sys
from datetime import datetime

class OnlyDebugFilter(logging.Filter):
    """
    日志过滤器：
    只允许 DEBUG 级别的日志通过。
    用于确保日志文件只包含我们的权重热力图数据，不包含 INFO/WARNING 等其他杂项。
    """
    def filter(self, record):
        return record.levelno == logging.DEBUG

def setup_logging():
    """
    配置全局日志系统
    
    输出流向：
    1. 控制台 (Console): 显示 INFO 及以上级别。用于看游戏进度。
    2. 文件 (File): 只显示 DEBUG 级别。用于存盘后的复盘分析 (权重热力图)。
       文件路径: data/YYYY-MM-DD_HH-MM-SS.txt
    """
    # 1. 确保 data 文件夹存在
    log_dir = "data"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 2. 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{timestamp}.txt"
    log_path = os.path.join(log_dir, log_filename)

    # 3. 获取根记录器
    logger = logging.getLogger("SmartHexLogger")
    logger.setLevel(logging.DEBUG)  # 根记录器捕获所有级别
    
    # 清除旧的处理器（防止重复添加导致日志重复）
    if logger.hasHandlers():
        logger.handlers.clear()

    # 4. 配置文件处理器 (写入 data/xxx.txt)
    # mode='w' 确保每次启动都是新文件 (虽然时间戳已保证这点)
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # [关键] 添加过滤器，确保文件中全是干货(DEBUG)，没有 INFO 干扰
    file_handler.addFilter(OnlyDebugFilter())
    
    # 简化格式，文件里直接看数据
    file_formatter = logging.Formatter('%(message)s') 
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 5. 配置控制台处理器 (输出到屏幕)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    # 控制台格式带上时间，方便看进度
    console_formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

# 初始化日志实例 (单例模式)
# 其他模块通过 from .utils import logger 使用
logger = setup_logging()


def count_empty_tiles(board) -> int:
    """
    统计棋盘上的空位数量
    """
    return sum(
        1 for x in range(board.size) 
        for y in range(board.size) 
        if board.tiles[x][y].colour is None
    )


def hash_board(board) -> str:
    """
    计算棋盘状态的字符串哈希。
    用于 MCTS 根节点复用和 Pattern 缓存键。
    
    格式: "0R0B..." (0=Empty, R=Red, B=Blue)
    """
    chars = []
    for x in range(board.size):
        for y in range(board.size):
            tile = board.tiles[x][y]
            if tile.colour is None:
                chars.append("0")
            elif tile.colour.value == 1: # 假设 RED value 是 1，需根据实际 Enum 调整，这里用通用逻辑
                 # 为了通用性，直接判断颜色对象
                 # 注意：这里需要确保 Colour 枚举转字符的一致性
                 # 简单起见，我们假设 0/1 索引或者 R/B 字符
                 chars.append("R" if str(tile.colour) == "Colour.RED" else "B") 
            else:
                 # 兜底
                 chars.append("X")
                 
    # 优化：如果是通过 value 判断 (更稳健)
    # 假设 Colour.RED.value = 1, Colour.BLUE.value = 2
    # 这里我们重写一个更通用的逻辑
    return "".join([
        "0" if t.colour is None else ("R" if str(t.colour).endswith("RED") else "B")
        for row in board.tiles for t in row
    ])