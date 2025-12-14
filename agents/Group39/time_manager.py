# time_manager.py
import time
from typing import Optional

from agents.Group39.phase import GamePhase
from src.Board import Board
from src.Colour import Colour


class TimeManager:
    """
    只统计“我方思考时间”的时间管理器（重要！）
    - used_time: 累计我方 make_move 内实际消耗
    - begin_move/end_move: 逐步累加
    """

    def __init__(self):
        self.total_time_limit = 295.0  # 留缓冲
        self.used_time = 0.0
        self._move_start: Optional[float] = None

    def start_timer(self):
        self.used_time = 0.0
        self._move_start = None

    def begin_move(self):
        # 防御：避免重复 begin 导致计时错乱
        if self._move_start is None:
            self._move_start = time.perf_counter()

    def end_move(self):
        if self._move_start is None:
            return
        self.used_time += (time.perf_counter() - self._move_start)
        self._move_start = None

    def _current_move_elapsed(self) -> float:
        """当前回合已经消耗了多少（尚未 end_move 累加的部分）"""
        if self._move_start is None:
            return 0.0
        return max(0.0, time.perf_counter() - self._move_start)

    def get_remaining_time(self) -> float:
        # 关键：要减去当前回合正在消耗的时间
        remaining = self.total_time_limit - self.used_time - self._current_move_elapsed()
        return max(0.0, remaining)

    def get_game_phase(self, board: Board) -> GamePhase:
        total = board.size * board.size
        empty = sum(
            1 for x in range(board.size)
            for y in range(board.size)
            if board.tiles[x][y].colour is None
        )
        filled = total - empty  # 双方总落子数（ply）

        OPENING_MY_MOVES = 3
        opening_end = 2 * OPENING_MY_MOVES  # 10 ply
        midgame_end = int(total * 0.60)  # 仍然用比例划后期

        if filled <= opening_end:
            return GamePhase.OPENING
        elif filled <= midgame_end:
            return GamePhase.MIDGAME
        else:
            return GamePhase.LATEGAME

    def assess_position_complexity(self, board: Board, threat_detector=None, my_colour=None) -> float:
        """
        0~1：越大越复杂
        - 空位比例
        - 双方最短路接近程度（越接近越复杂）
        - 威胁数量（可选）
        """
        total = board.size * board.size
        empty = sum(
            1 for x in range(board.size)
            for y in range(board.size)
            if board.tiles[x][y].colour is None
        )
        empty_ratio = empty / total if total > 0 else 0.0

        threat_factor = 0.0
        if threat_detector and my_colour:
            # 只做轻量：看对手是否有一手必胜点数量（越多越复杂）
            opp = Colour.opposite(my_colour)
            if hasattr(threat_detector, "immediate_win_moves"):
                lose_moves = threat_detector.immediate_win_moves(board, opp)
                threat_factor = min(1.0, len(lose_moves) / 3.0)
            else:
                # 兼容旧版 ThreatDetector：没有 immediate_win_moves 就不算这一项
                threat_factor = 0.0

        connectivity_factor = 0.0
        try:
            from .connectivity_evaluator import ConnectivityEvaluator
            ev = ConnectivityEvaluator()
            r = ev.shortest_path_cost(board, Colour.RED)
            b = ev.shortest_path_cost(board, Colour.BLUE)
            if r != float('inf') and b != float('inf'):
                # 平均成本越低，越接近决胜，越复杂
                avg = (r + b) / 2.0
                connectivity_factor = max(0.0, 1.0 - avg / board.size)
        except:
            pass

        # 空位越多分支越大；接近胜负/威胁越多也复杂
        comp = 0.55 * empty_ratio + 0.25 * connectivity_factor + 0.20 * threat_factor
        return max(0.0, min(1.0, comp))

    def assess_position_criticality(self, board: Board, threat_detector=None, my_colour=None) -> float:
        """
        1.0 ~ 2.5：越大越关键
        """
        if threat_detector and my_colour:
            if hasattr(threat_detector, "immediate_win_moves"):
                if threat_detector.immediate_win_moves(board, my_colour):
                    return 2.3
                opp = Colour.opposite(my_colour)
                if threat_detector.immediate_win_moves(board, opp):
                    return 2.0
            else:
                # 兼容旧版：跳过该项（避免调用旧 detect_immediate_threats 的语义坑）
                pass

        # 后期略增关键性
        phase = self.get_game_phase(board)
        if phase == GamePhase.LATEGAME:
            return 1.25
        return 1.0

    def allocate_time(self, board: Board, total_time_remaining: float, total_turns_remaining: int,
                      threat_detector=None, my_colour=None) -> float:
        if total_turns_remaining <= 0:
            total_turns_remaining = 1

        phase = self.get_game_phase(board)

        # --- 安全保留：保证永远不会被耗尽到 quickmove ---
        reserve = max(6.0, total_time_remaining * 0.05)  # 至少留 6s 或 5%
        spendable = max(0.0, total_time_remaining - reserve)

        base = spendable / total_turns_remaining

        complexity = self.assess_position_complexity(board, threat_detector, my_colour)
        criticality = self.assess_position_criticality(board, threat_detector, my_colour)

        # 时间因子：复杂度 0~1 映射到 0.75~1.65
        comp_factor = 0.75 + 0.90 * complexity

        budget = base * comp_factor * criticality

        # --- 分阶段硬限制（避免中盘过度挥霍）---
        if phase == GamePhase.OPENING:
            budget = min(budget, 5)  # 开局也要质量
        elif phase == GamePhase.MIDGAME:
            budget = min(budget, 10)
        else:
            budget = min(budget, 7)  # 允许关键时刻到 5s

        # --- 绝对上限：单步最多用剩余的 10%（防止烧光）---
        budget = min(budget, total_time_remaining * 0.10)

        # --- 保底：别低到没意义 ---
        return max(0.12, budget)
