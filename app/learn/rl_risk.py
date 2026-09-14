"""RL risk controller — tabular Q-learning agent that adapts the global risk
scale from realized equity outcomes.

State  : (trend, vol_state, drawdown_bucket, streak_bucket)   -> 54 states
Action : risk scale multiplier in {0.25, 0.5, 0.75, 1.0, 1.25}
Reward : equity log-return since last action, penalized for drawdown
Policy : epsilon-greedy with decaying epsilon; Q(s,a) updated on transition.
"""
import math, random

ACTIONS = [0.25, 0.5, 0.75, 1.0, 1.25]


class QRiskAgent:
    def __init__(self, alpha=0.15, gamma=0.9, eps=0.35, eps_min=0.05,
                 eps_decay=0.995):
        self.q = {}                    # state -> [q per action]
        self.alpha, self.gamma = alpha, gamma
        self.eps, self.eps_min, self.eps_decay = eps, eps_min, eps_decay
        self.prev_state = None
        self.prev_action = None        # index
        self.prev_equity = None
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
        if s not in self.q:
            # optimistic-ish init favoring moderate risk
            self.q[s] = [0.0, 0.001, 0.002, 0.002, 0.0]
        return self.q[s]

    # ---------- agent step ----------
    def act(self, regime, drawdown, streak, equity):
        s = self.encode(regime, drawdown, streak)

        # learn from the previous transition
        if self.prev_state is not None and self.prev_equity and equity > 0:
            r = math.log(equity / self.prev_equity)
            r -= 0.5 * max(0.0, drawdown - 0.05)     # drawdown penalty
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
