"""
工具函数模块
"""
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s'
)

logger = logging.getLogger(__name__)


def count_empty_tiles(board) -> int:
    """
    统计空位数量
    
    Args:
        board: 棋盘状态
    
    Returns:
        int: 空位数量
    """
    return sum(
        1 for x in range(board.size) 
        for y in range(board.size) 
        if board.tiles[x][y].colour is None
    )


def hash_board(board) -> str:
    """
    计算棋盘哈希值（用于根节点复用）
    
    Args:
        board: 棋盘状态
    
    Returns:
        str: 哈希值
    """
    board_str = ""
    for x in range(board.size):
        for y in range(board.size):
            tile = board.tiles[x][y]
            if tile.colour is None:
                board_str += "0"
            elif tile.colour.value == 0:  # RED
                board_str += "R"
            else:  # BLUE
                board_str += "B"
    return board_str

