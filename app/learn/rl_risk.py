"""RL risk controller — tabular Q-learning agent that adapts the global risk
scale from realized equity outcomes.

State  : (trend, vol_state, drawdown_bucket, streak_bucket)   -> 54 states
Action : risk scale multiplier in {0.0, 0.25, 0.5, 0.75, 1.0, 1.25}
         0.0 = SIT OUT. Sit-out means "skip discretionary probes / extra
         names" — NOT notional=0 on a cost-viable conviction trade. The risk
         manager floors the CONVICTION scale at 0.25 so the agent can never
         freeze the account shut on a structurally profitable signal; it only
         surfaces `rl_sit_out` so the orchestrator can skip the exploration
         block.
Reward : equity log-return since last action, credited to the risk actually
         taken and penalized for drawdown. CRUCIALLY, the learning update is
         only applied on intervals where there was a real opportunity to trade
         (a fill fired or a position was open). When every trade was blocked by
         the fee/cost gate, equity is flat because the DOOR WAS LOCKED, not
         because sitting out was wise — crediting sit-out there would freeze the
         account harder, so we skip the update entirely.
Policy : epsilon-greedy with decaying epsilon; Q(s,a) updated on transition.
"""
import math, random

ACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
CONVICTION_FLOOR = 0.25          # RL can never size a conviction trade below this


class QRiskAgent:
    def __init__(self, alpha=0.15, gamma=0.9, eps=0.35, eps_min=0.05,
                 eps_decay=0.995):
        self.q = {}                    # state -> [q per action]
        self.alpha, self.gamma = alpha, gamma
        self.eps, self.eps_min, self.eps_decay = eps, eps_min, eps_decay
        self.prev_state = None
        self.prev_action = None        # index
        self.prev_equity = None
        self.prev_tradable = False     # was last interval a real chance to trade?
        self.n_updates = 0
        self.last_reward = 0.0

    # ---------- state encoding ----------
    @staticmethod
    def encode(regime, drawdown, streak):
        trend = regime.get("trend", "sideways")
        vol = regime.get("vol_state", "normal")
        dd_b = 0 if drawdown < 0.03 else 1 if drawdown < 0.08 else 2
        st_b = 0 if streak == 0 else 1 if streak <= 2 else 2
        return (trend, vol, dd_b, st_b)

    def _qrow(self, s):
        row = self.q.get(s)
        if row is None:
            # optimistic-ish init favoring moderate risk; sit-out neutral
            row = [0.0005, 0.0, 0.001, 0.002, 0.002, 0.0]
            self.q[s] = row
        elif len(row) != len(ACTIONS):
            # migrate legacy 5-action rows (pre sit-out) — prepend sit-out arm
            row = [0.0005] + list(row)
            row = (row + [0.0] * len(ACTIONS))[:len(ACTIONS)]
            self.q[s] = row
        return row

    # ---------- agent step ----------
    def act(self, regime, drawdown, streak, equity, tradable=True):
        """Pick a risk scale for this interval.

        `tradable`: was the PREVIOUS interval one where trading was actually
        possible (a fill fired, or a position was open managing risk)? If not,
        flat equity tells us nothing about the sit-out decision — the fee/cost
        gate locked the door — so we DON'T run a learning update for it. This is
        what stops the agent from "learning" that sitting out is great simply
        because it was never allowed to try.
        """
        s = self.encode(regime, drawdown, streak)

        # learn from the previous transition — only if it was a real chance to
        # trade (otherwise crediting sit-out for "not losing" freezes us shut).
        if (self.prev_state is not None and self.prev_equity and equity > 0
                and self.prev_tradable):
            ret = math.log(equity / self.prev_equity)
            prev_scale = ACTIONS[self.prev_action]
            # Reward shaping:
            #  * P&L is credited to the risk actually taken — sitting out (0.0)
            #    neither earns nor loses, so a losing regime pushes Q toward it.
            #  * the drawdown penalty scales with risk pressed, so leaning in
            #    during a drawdown is punished while sitting it out is not.
            # NOTE: no blanket holding cost. We punish CHURN (a fill that lost
            # after costs shows up directly in `ret`), not a locked door.
            r = ret * (0.15 + 0.85 * prev_scale)     # credit P&L to risk taken
            r -= 0.5 * max(0.0, drawdown - 0.05) * prev_scale   # DD penalty ∝ risk
            self.last_reward = r
            row_p = self._qrow(self.prev_state)
            best_next = max(self._qrow(s))
            a = self.prev_action
            row_p[a] += self.alpha * (r + self.gamma * best_next - row_p[a])
            self.n_updates += 1
            self.eps = max(self.eps_min, self.eps * self.eps_decay)

        # choose next action (epsilon-greedy)
        row = self._qrow(s)
        if random.random() < self.eps:
            a = random.randrange(len(ACTIONS))
        else:
            mx = max(row)
            a = random.choice([i for i, v in enumerate(row) if v == mx])

        self.prev_state, self.prev_action, self.prev_equity = s, a, equity
        self.prev_tradable = tradable
        return ACTIONS[a]

    def stats(self):
        cur = None
        if self.prev_state is not None:
            row = self._qrow(self.prev_state)
            cur = {
                "state": "/".join(map(str, self.prev_state)),
                "chosen_scale": ACTIONS[self.prev_action],
                "q_values": {str(ACTIONS[i]): round(v, 6) for i, v in enumerate(row)},
            }
        return {
            "n_updates": self.n_updates,
            "epsilon": round(self.eps, 3),
            "states_visited": len(self.q),
            "last_reward": round(self.last_reward, 6),
            "current": cur,
        }


agent = QRiskAgent()
