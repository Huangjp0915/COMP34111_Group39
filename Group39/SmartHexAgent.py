"""
Group39 Hex AI Agent - 主代理类
EH-v4.5 (带有极致详细的注释，注释由ChatGPT总结撰写)
"""

# ========================
# 【这个 Agent 到底怎么做决策】
# 每回合 make_move() 会按下面顺序“早停”：
# (0) 时间极少 -> _get_quick_move(): 只看必须防/必赢点，绝不跑 MCTS
# (1) turn==1 -> _choose_balanced_opening(): 专门解决“第一手总是同一点/被 swap 针对”的问题
# (2) turn==2 -> _decide_swap(): 专门解决“我到底要不要 swap”的问题（更稳健的 2-ply）
# (3) ThreatDetector: 有必赢/必防 -> 直接下
# (4) 规则兜底：暴力“一步必杀/一步必防” -> 直接下（慢但可靠）
# (5) 防守补丁：对手明显更顺/快赢 -> 先挡一手
# (6) 正常走：分配 time_budget -> MCTS 搜索
# (7) 后处理：如果 MCTS 给了明显离谱点 -> 用轻量评分替换
# ========================

# ========================
# 【关于 shortest_path_cost 的解释】
# connectivity_evaluator.shortest_path_cost(board, colour)：
# - 这是一个“连通代价”，越小越接近连成一条边到另一条边（越接近赢）。
# - 所以我们常用 advantage = opp_cost - my_cost：
#     > opp_cost 大、my_cost 小 -> advantage 越大 -> 我越舒服。
# 注意：这里 advantage 不是胜率，只是“很粗的近似形势评分”。
# ========================

# ============================================================
# 可调参说明（给交接同学看的）
# ============================================================
#
# 这份 Agent 里有一些“经验阈值/权重”，它们不是数学定理，
# 但每一个都对应一个很具体的现象。改参数 = 改行为。
#
# -----------------------------
# A. 开局（_choose_balanced_opening）
# -----------------------------
#
# 1) 开局候选范围（目前写死在函数里）
#    dx, dy in [-2, 2]，并且 dist=|dx|+|dy| 取 1..3
#    - dist=0（正中心）刻意不允许：因为中心通常太强，会诱发对手 swap，
#      或者被 swap 后你吃亏（尤其你开局逻辑偏“保守抗swap”时）。
#    想更“中心派”：把 dist 允许到 0（但风险：swap 更频繁/更容易被针对）。
#
# 2) 2-ply 结构（逻辑固定，不建议乱改）
#    A: 对手不 swap -> 我下第一手 -> 对手用轻量策略回应 -> 评估 advA
#    B: 对手 swap    -> 我下第一手 -> 对手swap后身份互换 -> 评估 advB
#    最终用 worst = min(advA, advB) 做抗最坏情况。
#
# 3) center tie-break 系数（当前：0.03）
#    worst -= 0.03 * dist_to_center
#    - 调大：更偏向中心（更像人类腹地开局），但可能更容易诱发 swap
#    - 调小：更“抗swap”、更愿意离中心一点
#    推荐范围：0.01 ~ 0.06
#
# 4) 软随机 top-K（当前：K=5）
#    从分数最高的前 K 个开局点里抽样，而不是固定选第1名
#    - K 越小：越稳定，越“像背谱”，更容易每局固定点
#    - K 越大：越随机，越难针对，但可能略弱（会选到次优点）
#    推荐范围：3 ~ 7
#
# 5) 软随机温度 temp（当前：temp=0.35）
#    权重 w = exp((score - best_score)/temp)
#    - temp 越小：更接近“永远选第一名”（强但固定）
#    - temp 越大：更平均随机（不固定但可能更弱）
#    推荐范围：0.20 ~ 0.60
#    如果你观察到“总在同一点开局”：增大 temp 或增大 K
#
# 6) 开局对手回应/轻量选点候选规模 k_limit（当前：60）
#    - 越大：对手回应更像“聪明人”，评估更可信，但更慢
#    - 越小：更快，但对手回应可能很蠢，导致开局评估失真
#    推荐范围：40 ~ 90
#
#
# -----------------------------
# B. swap 决策（_decide_swap）
# -----------------------------
#
# 1) swap 2-ply margin（当前：margin=0.35）
#    判定：只有 advB > advA + margin 才 swap
#    - margin 越大：越不爱 swap（更保守，减少“老swap”）
#    - margin 越小：越爱 swap（更激进，可能更强也可能更像赌博）
#    推荐范围：0.25 ~ 0.80
#
#    怎么选：
#    - 你觉得“swap 太多/太激进”：把 margin 提高（比如 0.50 / 0.65 / 0.80）
#    - 你觉得“该swap不swap被开局压制”：把 margin 降低（比如 0.25 / 0.30）
#
# 2) swap 模拟最关键的“翻色归属”（不要改错！）
#    bB.set_tile_colour(opp_move.x, opp_move.y, opp_c)
#    含义：swap 后我方颜色变为 opp_c，因此第一手棋子归我后必须是 opp_c。
#    这一行写反会直接把 swap 判断搞废（严重 bug）。
#
# 3) swap 轻量选点 k_limit（当前：60）
#    同开局：越大越准越慢。推荐 40~90。
#
#
# -----------------------------
# C. 防守补丁（_choose_defensive_block）
# -----------------------------
#
# 1) 触发阈值 base_opp <= 3.2
#    - 代表“对手连通代价已经很低，接近赢”
#    - 调大：更早进入防守（更敏感，可能更像只防不攻）
#    - 调小：更晚防守（更贪，可能来不及）
#    推荐范围：2.8 ~ 3.8
#
# 2) 触发阈值 base_opp + 0.6 < base_my
#    - 代表“对手明显比我顺很多”（差距超过 0.6）
#    - 0.6 调大：更不容易触发防守；调小：更容易触发防守
#    推荐范围：0.4 ~ 1.0
#
# 3) 防守覆盖门槛 best_gain >= 0.55
#    - gain 表示“这一步对抬高对手代价有多明显”
#    - 门槛越大：越少强制防守（更相信 MCTS）
#    - 门槛越小：更频繁强制防守（更像脚本防守）
#    推荐范围：0.35 ~ 0.90
#
#
# -----------------------------
# D. MCTS 后处理（_postprocess_mcts_move）
# -----------------------------
#
# 1) 替换阈值：best_score > mcts_score + 0.65
#    - 越大：越不干预 MCTS（尊重搜索，但可能保留离谱点）
#    - 越小：越经常干预（更稳，但可能破坏 MCTS 全局计划）
#    推荐范围：0.45 ~ 0.90
#
# 2) light_score 权重 0.55 / 0.02（写在 _score_move_light）
#    score = (opp_cost提升) - 0.55*(my_cost变差) - 0.02*中心惩罚
#    - 0.55 越大：越怕自己走坏，风格更保守
#    - 0.55 越小：越偏“搞对手”，可能更激进
#    推荐范围：0.40 ~ 0.75
#
#    center_penalty 0.02：
#    - 越大：越偏中心，越像人类腹地；但也可能导致“都往中间挤”
#    推荐范围：0.00 ~ 0.05
#
#
# -----------------------------
# E. 典型现象 -> 对应调参
# -----------------------------
#
# 现象1：总在同一个点开局（比如老是 2列4行）
#   优先调：temp ↑ 或 K ↑
#   次选：候选范围扩大一点（比如 dist<=4），但注意会更慢
#
# 现象2：swap 太频繁（感觉老swap）
#   优先调：margin ↑（0.35 -> 0.50 -> 0.65 -> 0.80）
#   同时检查：advA/advB 是否总是极端（比如 -1/+1），那可能 evaluator 在小局面离散太粗
#
# 现象3：突然集体“降智”（两边都像乱下）
#   常见原因：
#   - 时间预算太小（time_budget不足导致 MCTS 树太浅）
#   - 轻量候选太少（k_limit太小导致“看不见好点”）
#   - 后处理阈值太低（0.65太小导致频繁覆盖 MCTS）
#
# 现象4：太保守，只防不攻
#   调：防守触发更严格（base_opp阈值↓、差距阈值↑、best_gain门槛↑）
#
# ============================================================

import copy
import random

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

from .mcts_engine import MCTSEngine
from .threat_detector import ThreatDetector
from .connectivity_evaluator import ConnectivityEvaluator
from .pattern_recognizer import PatternRecognizer
from .time_manager import TimeManager
from .utils import logger
from typing import Optional


class SmartHexAgent(AgentBase):
    """
    Group39 Hex AI代理（总控层）

    你可以把它理解成：
    - ThreatDetector：处理“必须马上下”的战术点（赢/防输）
    - ConnectivityEvaluator：提供“粗略形势评分”（shortest_path_cost）
    - MCTS：主要负责中后盘的全局规划
    - 本 SmartHexAgent：把这些模块按优先级串起来，并做一些保险补丁
    """
    def __init__(self, colour: Colour):
        super().__init__(colour)

        # --- 模块初始化 ---
        # 注意：colour 可能会因为 swap 改变，所以在处理 swap 时要同步更新模块里的 colour
        self.mcts_engine = MCTSEngine(colour)
        self.threat_detector = ThreatDetector(colour)
        self.connectivity_evaluator = ConnectivityEvaluator()
        self.pattern_recognizer = PatternRecognizer(colour)

        self.time_manager = TimeManager()
        self.time_manager.start_timer()

        # --- 状态变量 ---
        self.last_move = None
        self.turn_count = 0
        self.has_swapped = False   # 我方是否已经执行过 swap（执行过就不能再 swap）


    # ============================================================
    # 1) 规则层兜底：暴力检查“一步必杀/一步必防”
    # ============================================================
    def _find_winning_move_by_simulation(self, board: Board, colour: Colour) -> Optional[Move]:
        """
        【目的】
        用最笨但最可靠的方式找“一步就能赢”的落子。

        【为什么存在】
        - ThreatDetector 可能漏（或者实现上有 bug / 只覆盖某类威胁）
        - MCTS 也可能因为时间不足/随机性，错过“眼前必赢”
        所以这里做一个“终极保险”。

        【代价】
        - 会 deep copy + 遍历全盘空位 -> 很慢
        所以只在关键地方调用（不是每回合无脑调用很多次）。

        【返回】
        - 找到就返回 Move(x,y)
        - 找不到返回 None
        """
        size = board.size
        for x in range(size):
            for y in range(size):
                if board.tiles[x][y].colour is None:
                    test_board = copy.deepcopy(board)
                    test_board.set_tile_colour(x, y, colour)
                    if test_board.has_ended(colour):
                        return Move(x, y)
        return None

    def _check_priority_moves(self, board: Board) -> Optional[Move]:
        """
        【目的】
        显式规则层：只做两件事
        1) 我方是否有一步必赢？有 -> 直接赢
        2) 对手是否有一步必赢？有 -> 必须堵（否则下一手就输）

        【为什么它要在 MCTS 前】
        - 这是“战术必然”，不需要搜索
        - 让 AI 不会出现“明明能赢却去下别的”的离谱情况
        """
        # 1) 我方一步必赢
        win_move = self._find_winning_move_by_simulation(board, self.colour)
        if win_move is not None:
            return win_move

        # 2) 对手一步必赢 -> 我方必须防
        opp = self.opp_colour()
        block_move = self._find_winning_move_by_simulation(board, opp)
        if block_move is not None:
            return block_move

        return None


    # ============================================================
    # 2) 一些“局面判断小工具”
    # ============================================================
    def _is_board_empty(self, board: Board) -> bool:
        """
        棋盘是否完全为空（理论上 turn==1 就是空盘，但这里写成独立工具函数更保险）
        """
        size = board.size
        for x in range(size):
            for y in range(size):
                if board.tiles[x][y].colour is not None:
                    return False
        return True

    def _is_my_first_move(self, board: Board) -> bool:
        """
        【目的】
        判断这是不是“我方在本局的第一颗棋”。

        【为什么不是用 turn 判断】
        因为 Hex 有 swap：
        - 你可能在 turn==2 才第一次真正落子（例如你选择 swap）
        - 或者你本来是先手，但对手 swap 后身份变了
        所以这里只看棋盘上有没有我方颜色的棋子，更稳。
        """
        size = board.size
        for x in range(size):
            for y in range(size):
                if board.tiles[x][y].colour == self.colour:
                    return False
        return True


    # ============================================================
    # 3) 开局：_choose_balanced_opening（详细解释版）
    # ============================================================
    def _choose_balanced_opening(self, board: Board) -> Move:
        """
        【目的】
        解决两个典型开局问题：
        1) “每盘都下同一个点” -> 会被背谱针对，看起来像降智
        2) “第一手太强/太中心” -> 对手 swap 后我会吃亏

        【核心思想：2-ply worst-case + 软随机】
        我考虑每个候选开局点 (x,y)：
        - 情况A：对手不 swap，会下一个“轻量最优回应”（我们用局部候选+cost增量近似）
        - 情况B：对手 swap（不落子，但第一手棋子归对手；我方颜色互换）
        然后我取 worst = min(advA, advB)，让这个 worst 最大。
        => 这就是“抗针对/抗 swap”的开局。

        【软随机是什么】
        - 我不是永远选分数最高的那个点
        - 而是从 top-K 里面按权重抽一个
        - 权重越高越偏向最好点，但仍然有小概率选到次优点
        => 这样对手很难靠固定套路针对你

        【候选范围】
        - 只看中心附近（dx,dy 在 [-2,2]）
        - 但排除 dist==0（正中心），因为中心往往过强，swap 风险更大
        - dist 限制在 1..3 是为了：不要太偏边、也不要离中心太远没意义
        """
        size = board.size
        c = size // 2
        my_c = self.colour
        opp_c = self.opp_colour()

        def advantage(b: Board, me: Colour) -> float:
            """
            优势定义（统一口径）：
            advantage = opp_cost - my_cost
            - my_cost 越小越好
            - opp_cost 越大越好
            所以 advantage 越大越“我更舒服”
            """
            oc = Colour.BLUE if me == Colour.RED else Colour.RED
            my_cost = self.connectivity_evaluator.shortest_path_cost(b, me)
            opp_cost = self.connectivity_evaluator.shortest_path_cost(b, oc)
            return opp_cost - my_cost

        def pick_best_reply_light(b: Board, player: Colour, k_limit: int = 60) -> Optional[Move]:
            """
            【目的】
            模拟“对手在不 swap 情况下，会怎么回应我第一手”。

            【方法：轻量近似，不跑 MCTS】
            1) 生成局部候选点（靠近棋子 + 中心附近）
            2) 对每个候选点，假设 player 下在那里
            3) 看它对 (opp_cost - my_cost) 的增量贡献
            4) 选贡献最大的点

            【为什么 k_limit 默认 60】
            - 太小：对手回应可能选得很蠢（评估失真）
            - 太大：每个候选都要 deep copy + 两次 shortest_path_cost，太慢
            60 是一个“够用但不爆炸”的经验值（你也可以调参）。
            """
            cand = set(self._generate_local_candidates(b, player, k_limit=k_limit))
            cand.update(self._generate_local_candidates(
                b, Colour.BLUE if player == Colour.RED else Colour.RED, k_limit=k_limit
            ))

            base_my = self.connectivity_evaluator.shortest_path_cost(b, player)
            base_opp = self.connectivity_evaluator.shortest_path_cost(
                b, Colour.BLUE if player == Colour.RED else Colour.RED
            )

            best_mv, best_sc = None, -1e18
            for x, y in cand:
                if b.tiles[x][y].colour is not None:
                    continue
                b2 = copy.deepcopy(b)
                b2.set_tile_colour(x, y, player)

                new_my = self.connectivity_evaluator.shortest_path_cost(b2, player)
                new_opp = self.connectivity_evaluator.shortest_path_cost(
                    b2, Colour.BLUE if player == Colour.RED else Colour.RED
                )

                # sc 越大说明这步越能“让对手更难 + 让自己更易”
                # 0.6 是一个权重：更强调“别把自己走坏”
                sc = (new_opp - base_opp) - 0.6 * (new_my - base_my)

                if sc > best_sc:
                    best_sc, best_mv = sc, Move(x, y)

            return best_mv

        # --- 构造候选开局点（中心附近，不要正中心） ---
        cand = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                x, y = c + dx, c + dy
                if 0 <= x < size and 0 <= y < size and board.tiles[x][y].colour is None:
                    dist = abs(dx) + abs(dy)
                    if 1 <= dist <= 3:
                        cand.append((x, y))

        if not cand:
            return self._get_first_valid_move(board)

        scored = []
        for x, y in cand:
            # ---------- 情况A：对手不swap ----------
            # 我方先下 (x,y)
            bA = copy.deepcopy(board)
            bA.set_tile_colour(x, y, my_c)

            # 对手选择一个“轻量最优回应”
            opp_reply = pick_best_reply_light(bA, opp_c)
            if opp_reply is not None:
                bA.set_tile_colour(opp_reply.x, opp_reply.y, opp_c)

            # 评估我方在“不swap且对手回应后”的优势
            advA = advantage(bA, my_c)

            # ---------- 情况B：对手swap ----------
            # 注意 swap 的关键点（写清楚避免以后改错）：
            #
            # - swap 发生后：我方颜色变成 opp_c
            # - 但是棋盘上 (x,y) 这颗棋仍然是 my_c
            #   因为它“归对手”了（对手拿走我第一手的棋子）
            #
            # 所以这里棋盘不用翻色，只需要用“新身份颜色”去评估优势。
            bB = copy.deepcopy(board)
            bB.set_tile_colour(x, y, my_c)
            advB = advantage(bB, opp_c)  # 我方视角换成 opp_c

            # worst-case：对手会选让我最难受的那条分支
            worst = min(advA, advB)

            # tie-break：轻微偏向中心（但不要压过抗swap的主目标）
            worst -= 0.03 * (abs(x - c) + abs(y - c))

            scored.append((worst, x, y))

        # --- 软随机：top-K 里加权抽样 ---
        scored.sort(key=lambda t: t[0], reverse=True)
        K = min(5, len(scored))
        top = scored[:K]

        # temp 解释：
        # - temp 越小：越接近“永远选第一名”（更强但更固定）
        # - temp 越大：越随机（更难被针对但可能略弱）
        temp = 0.35
        weights = [pow(2.71828, (s - top[0][0]) / temp) for s, _, _ in top]

        _, bx, by = random.choices(top, weights=weights, k=1)[0]
        return Move(bx, by)


    # ============================================================
    # 4) 局部候选点生成（解释为什么能减少“固定下同一点”的问题）
    # ============================================================
    def _generate_local_candidates(self, board: Board, focus_colour: Colour, k_limit: int = 40):
        """
        【目的】
        生成“看起来更可能有意义”的候选落点，而不是全盘枚举。

        候选来源：
        1) focus_colour 的所有棋子周围一圈空位（局部战斗相关）
        2) 棋盘中心附近一小块（避免完全陷入局部、也避免开局被边缘诱导）

        为什么会出现“总是同一点”的现象？
        - 如果候选集太小 + 评分函数是确定性的 + tie-break 固定
        => 那么每局同样的局面会选同样的点
        所以开局这里我们又加了“软随机”来避免固定。

        k_limit：
        - 太小：候选集不够，容易错过好点
        - 太大：后面会对每个候选做 deepcopy+cost 计算，时间爆炸
        """
        size = board.size
        c = size // 2
        cand = set()

        # (1) focus_colour 棋子邻域空位
        for x in range(size):
            for y in range(size):
                if board.tiles[x][y].colour == focus_colour:
                    for nx in range(max(0, x - 1), min(size, x + 2)):
                        for ny in range(max(0, y - 1), min(size, y + 2)):
                            if board.tiles[nx][ny].colour is None:
                                cand.add((nx, ny))

        # (2) 中心附近空位
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                x, y = c + dx, c + dy
                if 0 <= x < size and 0 <= y < size and board.tiles[x][y].colour is None:
                    cand.add((x, y))

        # (3) 只取最靠近中心的前 k_limit 个（让候选稳定、可控）
        cand_list = list(cand)
        cand_list.sort(key=lambda xy: abs(xy[0] - c) + abs(xy[1] - c))
        return cand_list[:k_limit]


    # ============================================================
    # 5) 防守补丁（解释阈值意义）
    # ============================================================
    def _choose_defensive_block(self, board: Board) -> Optional[Move]:
        """
        【目的】
        有些 AI 会出现“只顾着推进自己线路，不管对手已经快连成了”。
        这个函数就是专门防这个的：当对手明显快赢时，强制优先挡一手。

        触发条件：
        - base_opp <= 3.2 ：对手“连通代价很低”，离赢很近（阈值经验值）
        - 或 base_opp + 0.6 < base_my ：对手比我顺很多（差距明显）

        选点策略：
        - 生成靠近对手棋子的候选点
        - 我方假设下在该点
        - 看对手 cost 被抬高多少（越高越好）
        - 同时轻微惩罚“把自己也堵死”的点

        覆盖条件：
        - best_gain >= 0.55 才覆盖
          否则容易变成“过度防守”，该攻的时候也在瞎挡。
        """
        my_c = self.colour
        opp_c = self.opp_colour()

        try:
            base_my = self.connectivity_evaluator.shortest_path_cost(board, my_c)
            base_opp = self.connectivity_evaluator.shortest_path_cost(board, opp_c)

            if not (base_opp <= 3.2 or base_opp + 0.6 < base_my):
                return None

            candidates = self._generate_local_candidates(board, opp_c, k_limit=50)
            if not candidates:
                return None

            best = None
            best_gain = -1e9

            for x, y in candidates:
                b2 = copy.deepcopy(board)
                b2.set_tile_colour(x, y, my_c)

                new_opp = self.connectivity_evaluator.shortest_path_cost(b2, opp_c)
                new_my = self.connectivity_evaluator.shortest_path_cost(b2, my_c)

                # gain 大：说明这步显著“抬高对手代价”
                # -0.35*(new_my-base_my)：防止把自己堵死
                gain = (new_opp - base_opp) - 0.35 * (new_my - base_my)

                if gain > best_gain:
                    best_gain = gain
                    best = Move(x, y)

            if best and best_gain >= 0.55 and self._is_valid_move(best, board):
                return best
            return None

        except Exception as e:
            logger.error(f"Error in defensive block: {e}")
            return None


    # ============================================================
    # 6) 轻量评分 + MCTS 后处理（解释阈值）
    # ============================================================
    def _score_move_light(self, board: Board, move: Move) -> float:
        """
        【用途】
        给一个普通落子打一个“很快的粗分数”，用于：
        - MCTS 后处理 sanity check：MCTS 是否下了明显离谱点

        评分结构（越大越好）：
        + (opp_cost 上升量)      -> 我让对手更难连通
        - 0.55*(my_cost 上升量)  -> 但别让我自己也变难连通
        - center_penalty         -> 很轻微偏好中心（只是 tie-break）

        0.55 是经验权重：
        - 过大：会太保守，只顾自己路径
        - 过小：会太激进，只顾恶心对手
        """
        if move.x == -1 and move.y == -1:
            return -1e9

        my_c = self.colour
        opp_c = self.opp_colour()

        base_my = self.connectivity_evaluator.shortest_path_cost(board, my_c)
        base_opp = self.connectivity_evaluator.shortest_path_cost(board, opp_c)

        b2 = copy.deepcopy(board)
        b2.set_tile_colour(move.x, move.y, my_c)

        new_my = self.connectivity_evaluator.shortest_path_cost(b2, my_c)
        new_opp = self.connectivity_evaluator.shortest_path_cost(b2, opp_c)

        size = board.size
        c = size // 2
        center_penalty = 0.02 * (abs(move.x - c) + abs(move.y - c))

        return (new_opp - base_opp) - 0.55 * (new_my - base_my) - center_penalty

    def _postprocess_mcts_move(self, board: Board, move: Move) -> Move:
        """
        【目的】
        MCTS 偶尔会因为随机 playout / 噪声 / 树没长够 给出“很怪的点”。
        这里做轻量修正，但非常克制，避免破坏 MCTS 原本计划。

        做法：
        - 计算 MCTS 给的 move 的 light_score
        - 在局部候选集合里找一个 best_score
        - 只有当 best_score > mcts_score + 0.65 才替换
          0.65 是“明显更好才改”的阈值（经验值）
        """
        try:
            if move is None or (move.x != -1 and board.tiles[move.x][move.y].colour is not None):
                return move

            mcts_score = self._score_move_light(board, move)

            cand = set(self._generate_local_candidates(board, self.colour, k_limit=35))
            cand.update(self._generate_local_candidates(board, self.opp_colour(), k_limit=35))

            best_move = move
            best_score = mcts_score

            for x, y in cand:
                mv = Move(x, y)
                if not self._is_valid_move(mv, board):
                    continue
                sc = self._score_move_light(board, mv)
                if sc > best_score:
                    best_score = sc
                    best_move = mv

            if best_move != move and best_score > mcts_score + 0.65:
                logger.info(f"Postprocess override: {move} -> {best_move} (score {mcts_score:.3f} -> {best_score:.3f})")
                return best_move

            return move

        except Exception as e:
            logger.error(f"Error in MCTS postprocess: {e}")
            return move


    # ============================================================
    # 7) make_move：主决策入口（每一步为什么放这里）
    # ============================================================
    def make_move(self, turn: int, board: Board, opp_move: Optional[Move]) -> Move:
        """
        主决策函数（每回合调用一次）

        参数说明：
        - turn：回合数（一般 turn==1 是空盘第一手）
        - board：当前棋盘
        - opp_move：对手上一手
            * None：说明我先手第一手
            * is_swap()：说明对手刚 swap
        """
        self.turn_count = turn

        try:
            # (0) 时间安全：快没时间就别想太多，先保证下出合法棋 + 不被一招秒
            remaining_time = self.time_manager.get_remaining_time()
            if remaining_time < 0.05:
                return self._get_quick_move(board)

            # (1) 第一手开局：使用“抗swap+抗回应”的开局点（并且带软随机）
            if turn == 1:
                return self._choose_balanced_opening(board)

            # (2) 如果对手刚刚 swap：引擎会改 self.colour，但我们要同步所有模块
            if opp_move and opp_move.is_swap():
                self.has_swapped = True

                # 同步颜色（否则模块还以为自己是旧颜色，会直接算错）
                self.mcts_engine.colour = self.colour
                self.threat_detector.colour = self.colour
                self.pattern_recognizer.colour = self.colour

                # swap 后局面“身份”变了，旧树没意义，直接清掉
                self.mcts_engine.root_node = None
                self.mcts_engine.last_move = None
                self.mcts_engine.last_board_hash = None

            # (3) turn==2：我是否要 swap（只在我还没 swap 且对手没 swap 时）
            if turn == 2 and not self.has_swapped and (not opp_move or not opp_move.is_swap()):
                swap_move = self._decide_swap(board, opp_move)
                if swap_move.x == -1 and swap_move.y == -1:
                    self.has_swapped = True
                return swap_move

            # (4) ThreatDetector：优先级高于 MCTS（战术点不能错）
            immediate_threats = self.threat_detector.detect_immediate_threats(board, self.opp_colour())
            for x, y, threat_type in immediate_threats:
                if threat_type == "WIN":
                    return Move(x, y)
                elif threat_type == "LOSE":
                    return Move(x, y)

            # (5) 规则兜底：一步必赢/一步必防
            priority_move = self._check_priority_moves(board)
            if priority_move is not None:
                return priority_move

            # (6) 防守补丁：对手太顺先挡一手
            defensive = self._choose_defensive_block(board)
            if defensive is not None:
                return defensive

            # (7) 正常走：时间分配 -> MCTS
            total_turns_remaining = self._estimate_remaining_turns(board)
            time_budget = self.time_manager.allocate_time(
                board, remaining_time, total_turns_remaining, self.threat_detector, self.opp_colour()
            )

            # 根节点复用依赖 opp_move：对手正常落子才有意义
            if opp_move and not opp_move.is_swap():
                self.mcts_engine.last_move = opp_move
            else:
                self.mcts_engine.last_move = None
                self.mcts_engine.root_node = None
                self.mcts_engine.last_board_hash = None

            move = self.mcts_engine.search(board, time_budget)

            # (8) 合法性强制兜底：无论如何不能返回非法点
            if not move or not self._is_valid_move(move, board):
                return self._get_fallback_move(board)

            # (9) 后处理：只有“明显更好”才替换，避免瞎改破坏 MCTS
            move = self._postprocess_mcts_move(board, move)
            if not self._is_valid_move(move, board):
                return self._get_fallback_move(board)

            self.last_move = move
            return move

        except Exception as e:
            logger.error(f"Error in make_move: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_move(board)


    # ============================================================
    # 8) swap 判定（重点：把“翻色/归属”写得非常清楚）
    # ============================================================
    def _decide_swap(self, board: Board, opp_move: Optional[Move]) -> Move:
        """
        更科学的 swap 判定：超轻量 2-ply 对比

        ========================
        【swap 到底发生了什么？】
        ========================
        假设当前我是 my_c，对手是 opp_c（并且 opp_move 是对手第一手的位置）

        如果我选择 SWAP：
        1) 我方颜色变成 opp_c
        2) 对手颜色变成 my_c
        3) 棋盘上 opp_move 那颗棋子“归我”：
           -> 也就是那格颜色要变成 opp_c（我 swap 后的新颜色）

        这一步非常容易写反，所以这里明确写：
        bB.set_tile_colour(opp_move.x, opp_move.y, opp_c)

        ========================
        【2-ply 对比怎么做？】
        ========================
        A) 不 swap：
           - 我先下一步（用 pick_best_move_light 选轻量最优点）
           - 然后评估 advA

        B) swap：
           - 先把第一手棋子改成“归我（新颜色 opp_c）”
           - 然后轮到对手走一步（对手颜色是原 my_c）
           - 再评估 advB（此时我方颜色是 opp_c）

        最后：
        - 只有 advB > advA + margin 才 swap
        - margin 是“抑制过度 swap”的门槛：越大越保守
        """
        if self.has_swapped:
            return self._get_first_valid_move(board)

        if not opp_move or opp_move.is_swap():
            return self._get_first_valid_move(board)

        def opp_of(c: Colour) -> Colour:
            return Colour.BLUE if c == Colour.RED else Colour.RED

        def advantage(b: Board, my_colour: Colour) -> float:
            oc = opp_of(my_colour)
            my_cost = self.connectivity_evaluator.shortest_path_cost(b, my_colour)
            opp_cost = self.connectivity_evaluator.shortest_path_cost(b, oc)
            return opp_cost - my_cost

        def pick_best_move_light(b: Board, player_colour: Colour, k_limit: int = 60) -> Optional[Move]:
            """
            轻量选点（不跑 MCTS）：
            - 候选集来自局部生成器（双方棋子周围 + 中心附近）
            - 评分：抬高对手代价 + 降低自己代价（权重 0.6）
            """
            cand = set(self._generate_local_candidates(b, player_colour, k_limit=k_limit))
            cand.update(self._generate_local_candidates(b, opp_of(player_colour), k_limit=k_limit))

            base_my = self.connectivity_evaluator.shortest_path_cost(b, player_colour)
            base_opp = self.connectivity_evaluator.shortest_path_cost(b, opp_of(player_colour))

            best_mv = None
            best_score = -1e18

            for x, y in cand:
                if b.tiles[x][y].colour is not None:
                    continue
                b2 = copy.deepcopy(b)
                b2.set_tile_colour(x, y, player_colour)

                new_my = self.connectivity_evaluator.shortest_path_cost(b2, player_colour)
                new_opp = self.connectivity_evaluator.shortest_path_cost(b2, opp_of(player_colour))

                score = (new_opp - base_opp) - 0.6 * (new_my - base_my)

                if score > best_score:
                    best_score = score
                    best_mv = Move(x, y)

            return best_mv

        try:
            my_c = self.colour
            opp_c = self.opp_colour()

            # -------- A) 不 swap：我先走一步 --------
            bA = copy.deepcopy(board)
            mvA = pick_best_move_light(bA, my_c)
            if mvA is not None:
                bA.set_tile_colour(mvA.x, mvA.y, my_c)
            advA = advantage(bA, my_c)

            # -------- B) swap：第一手棋子归我（新颜色 opp_c），然后对手走一步 --------
            bB = copy.deepcopy(board)

            # 关键行：swap 后那颗棋子要变成我的新颜色（opp_c）
            bB.set_tile_colour(opp_move.x, opp_move.y, opp_c)

            # swap 后轮到对手走，对手颜色是原 my_c
            mvB_opp = pick_best_move_light(bB, my_c)
            if mvB_opp is not None:
                bB.set_tile_colour(mvB_opp.x, mvB_opp.y, my_c)

            # swap 后我方颜色 = opp_c
            advB = advantage(bB, opp_c)

            # margin：swap 门槛（越大越不爱 swap）
            margin = 0.35
            if advB > advA + margin:
                self.has_swapped = True
                logger.info(f"Agent decided to SWAP (advA={advA:.3f}, advB={advB:.3f}, margin={margin:.2f})")
                return Move(-1, -1)

            return self._get_first_valid_move(board)

        except Exception as e:
            logger.error(f"Error in swap decision: {e}")
            return self._get_first_valid_move(board)


    # ============================================================
    # 9) 兜底走法（详细解释每个 fallback 的意义）
    # ============================================================
    def _get_fallback_move(self, board: Board) -> Move:
        """
        【目的】
        无论发生什么异常，都要返回合法走法（比赛环境最重要：不能崩）

        fallback 顺序：
        1) 如果 MCTS root_node 与当前棋盘 hash 一致 -> 选 visits 最大的子节点（最可靠）
        2) PatternRecognizer 给出候选 -> 选权重最高且合法的
        3) 最后兜底：离中心最近的空位
        """
        try:
            if self.mcts_engine.root_node:
                from .utils import hash_board
                root_hash = hash_board(self.mcts_engine.root_node.board)
                current_hash = hash_board(board)
                if root_hash == current_hash and self.mcts_engine.root_node.children:
                    best_child = max(self.mcts_engine.root_node.children, key=lambda c: c.visits)
                    if best_child.move and self._is_valid_move(best_child.move, board):
                        if best_child.move.x != -1 and best_child.move.y != -1:
                            if board.tiles[best_child.move.x][best_child.move.y].colour is None:
                                return best_child.move

            patterns = self.pattern_recognizer.detect_simple_patterns(board, self.colour)
            if patterns:
                patterns.sort(key=lambda p: p[2], reverse=True)
                for x, y, _ in patterns:
                    move = Move(x, y)
                    if self._is_valid_move(move, board) and board.tiles[x][y].colour is None:
                        return move

            return self._get_first_valid_move(board)

        except Exception as e:
            logger.error(f"Error in fallback move: {e}")
            return self._get_first_valid_move(board)

    def _get_quick_move(self, board: Board) -> Move:
        """
        时间不足时的快速走法：
        - 只检查 immediate threats（必防/必赢）
        - 否则下一个最简单的合法点
        """
        try:
            immediate_threats = self.threat_detector.detect_immediate_threats(board, self.opp_colour())
            if immediate_threats:
                for x, y, threat_type in immediate_threats:
                    move = Move(x, y)
                    if self._is_valid_move(move, board):
                        return move
            return self._get_first_valid_move(board)

        except Exception as e:
            logger.error(f"Error in quick move: {e}")
            return self._get_first_valid_move(board)

    def _get_first_valid_move(self, board: Board) -> Move:
        """
        最终兜底：永远返回一个合法点（离中心最近的空位）

        为什么用“离中心最近”：
        - Hex 中心通常更有潜力（连接两边更灵活）
        - 作为兜底比随机乱下好
        """
        size = board.size
        center = size // 2
        best = None
        best_dist = 10 ** 9

        for x in range(size):
            for y in range(size):
                if board.tiles[x][y].colour is None:
                    dist = abs(x - center) + abs(y - center)
                    if dist < best_dist:
                        best_dist = dist
                        best = (x, y)

        if best is None:
            return Move(-1, -1)
        return Move(best[0], best[1])

    def _is_valid_move(self, move: Move, board: Board) -> bool:
        """
        走法合法性：
        - (-1,-1) 代表 swap：永远合法
        - 普通点：必须在棋盘内且为空
        """
        if move.x == -1 and move.y == -1:
            return True

        if 0 <= move.x < board.size and 0 <= move.y < board.size:
            return board.tiles[move.x][move.y].colour is None

        return False

    def _estimate_remaining_turns(self, board: Board) -> int:
        """
        估算剩余回合数（只用于时间管理，不要求精确）
        - empty_tiles / 2：因为双方轮流下
        """
        empty_tiles = sum(
            1 for x in range(board.size)
            for y in range(board.size)
            if board.tiles[x][y].colour is None
        )
        return max(1, empty_tiles // 2)