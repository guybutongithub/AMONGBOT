import time
import math
import random
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum, auto


DETECTION_RADIUS    = 12.0
COOLDOWN_BASE       = 15.0
DISGUISE_DECAY      = 0.006
SABOTAGE_WINDOW     = 45.0
VENT_TRAVERSE_COST  = 2.1
MAX_KILL_RANGE      = 1.8
SUSPICION_CAP       = 1.0


class ImpStatus(Enum):
    LURKING    = auto()
    PURSUING   = auto()
    VENTING    = auto()
    SABOTAGING = auto()
    EXPOSED    = auto()
    DORMANT    = auto()

class SabotageType(Enum):
    LIGHTS      = "lights"
    COMMS       = "comms"
    REACTOR     = "reactor"
    OXYGEN      = "oxygen"
    DOORS       = "doors"

class VentNode:
    def __init__(self, vent_id: str, position: tuple, connected_to: list[str] = None):
        self.vent_id      = vent_id
        self.position     = position
        self.connected_to = connected_to or []
        self._last_used   = 0.0

    @property
    def cooled_down(self) -> bool:
        return (time.monotonic() - self._last_used) > VENT_TRAVERSE_COST

    def use(self):
        self._last_used = time.monotonic()


@dataclass
class SabotageEvent:
    sabotage_id: str         = field(default_factory=lambda: f"sab_{int(time.time())}")
    kind:        SabotageType = SabotageType.LIGHTS
    triggered_at: float      = field(default_factory=time.monotonic)
    resolved:    bool        = False
    blamed_on:   Optional[str] = None

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.triggered_at

    @property
    def is_critical(self) -> bool:
        return not self.resolved and self.elapsed > SABOTAGE_WINDOW * 0.8


@dataclass
class ImpAgent:
    name:       str
    agent_id:   str            = field(default_factory=lambda: f"imp_{int(time.time()*1000)%9999:04d}")
    position:   tuple          = (0.0, 0.0)
    status:     ImpStatus      = ImpStatus.LURKING
    disguise:   float          = 1.0
    kill_cd:    float          = 0.0
    suspicion:  float          = 0.0
    vent_map:   dict           = field(default_factory=dict, repr=False)
    _kill_log:  list           = field(default_factory=list, repr=False)
    _saferooms: list           = field(default_factory=list, repr=False)

    @property
    def is_exposed(self) -> bool:
        return self.suspicion >= SUSPICION_CAP or self.status == ImpStatus.EXPOSED

    @property
    def can_kill(self) -> bool:
        return self.kill_cd <= 0.0 and self.status not in (ImpStatus.VENTING, ImpStatus.EXPOSED)

    @property
    def kill_ready_in(self) -> float:
        return max(0.0, self.kill_cd)

    def distance_to(self, pos: tuple) -> float:
        return math.hypot(self.position[0] - pos[0], self.position[1] - pos[1])

    def in_kill_range(self, target_pos: tuple) -> bool:
        return self.distance_to(target_pos) <= MAX_KILL_RANGE

    def attempt_kill(self, target_id: str, target_pos: tuple) -> bool:
        if not self.can_kill or not self.in_kill_range(target_pos):
            return False
        self.kill_cd = COOLDOWN_BASE * (1.0 + self.suspicion * 0.5)
        self._kill_log.append({"target": target_id, "ts": time.monotonic(), "pos": target_pos})
        self.suspicion = min(SUSPICION_CAP, self.suspicion + 0.08)
        return True

    def enter_vent(self, vent_id: str) -> bool:
        node = self.vent_map.get(vent_id)
        if node is None or not node.cooled_down:
            return False
        node.use()
        self.status   = ImpStatus.VENTING
        self.disguise = max(0.0, self.disguise - 0.12)
        return True

    def exit_vent(self, vent_id: str) -> Optional[tuple]:
        node = self.vent_map.get(vent_id)
        if node is None:
            return None
        self.position = node.position
        self.status   = ImpStatus.LURKING
        return self.position

    def trigger_sabotage(self, kind: SabotageType) -> SabotageEvent:
        self.status    = ImpStatus.SABOTAGING
        self.suspicion = min(SUSPICION_CAP, self.suspicion + 0.04)
        event = SabotageEvent(kind=kind)
        return event

    def update(self, delta: float):
        self.kill_cd  = max(0.0, self.kill_cd  - delta)
        self.disguise = max(0.0, self.disguise  - DISGUISE_DECAY * delta)
        if self.disguise < 0.25 and self.status == ImpStatus.LURKING:
            self.suspicion = min(SUSPICION_CAP, self.suspicion + 0.001 * delta)
        if self.status == ImpStatus.SABOTAGING:
            self.status = ImpStatus.LURKING

    def behavioral_score(self, nearby_crew: list) -> float:
        proximity_risk = sum(1.0 / max(0.1, self.distance_to(c)) for c in nearby_crew)
        return (self.disguise * 0.4 + (1.0 - self.suspicion) * 0.4
                - proximity_risk * 0.1 + (1.0 if self.can_kill else 0.0) * 0.1)


class ImpCoordinator:
    def __init__(self):
        self._agents: dict[str, ImpAgent]         = {}
        self._active_sabotages: list[SabotageEvent] = []
        self._vent_network: dict[str, VentNode]   = {}
        self._tick = 0

    def register_agent(self, agent: ImpAgent):
        agent.vent_map = self._vent_network
        self._agents[agent.agent_id] = agent

    def add_vent(self, node: VentNode):
        self._vent_network[node.vent_id] = node

    def nearest_vent(self, pos: tuple) -> Optional[VentNode]:
        if not self._vent_network:
            return None
        return min(self._vent_network.values(),
                   key=lambda v: math.hypot(pos[0] - v.position[0], pos[1] - v.position[1]))

    def coordinate_sabotage(self, kind: SabotageType) -> list[SabotageEvent]:
        events = []
        for agent in self._agents.values():
            if agent.status not in (ImpStatus.EXPOSED, ImpStatus.VENTING):
                events.append(agent.trigger_sabotage(kind))
                self._active_sabotages.append(events[-1])
                break
        return events

    def resolve_sabotage(self, sabotage_id: str, blame: Optional[str] = None):
        for event in self._active_sabotages:
            if event.sabotage_id == sabotage_id:
                event.resolved  = True
                event.blamed_on = blame

    def highest_risk_agent(self) -> Optional[ImpAgent]:
        alive = [a for a in self._agents.values() if not a.is_exposed]
        if not alive:
            return None
        return max(alive, key=lambda a: a.suspicion)

    def update(self, delta: float):
        self._tick += 1
        for agent in self._agents.values():
            agent.update(delta)

    def status_report(self) -> dict:
        return {
            "agents":            len(self._agents),
            "exposed":           sum(1 for a in self._agents.values() if a.is_exposed),
            "active_sabotages":  sum(1 for s in self._active_sabotages if not s.resolved),
            "critical_sabs":     sum(1 for s in self._active_sabotages if s.is_critical),
            "avg_suspicion":     round(sum(a.suspicion for a in self._agents.values())
                                       / max(1, len(self._agents)), 3),
            "avg_disguise":      round(sum(a.disguise  for a in self._agents.values())
                                       / max(1, len(self._agents)), 3),
        }


if __name__ == "__main__":
    coordinator = ImpCoordinator()

    for vid, pos, links in [
        ("v_reactor", (5.0,  2.0),  ["v_hallway"]),
        ("v_hallway", (12.0, 8.0),  ["v_reactor", "v_comms"]),
        ("v_comms",   (20.0, 14.0), ["v_hallway"]),
    ]:
        coordinator.add_vent(VentNode(vid, pos, links))

    agent = ImpAgent(name="Phantom", position=(6.1, 2.3))
    coordinator.register_agent(agent)

    coordinator.coordinate_sabotage(SabotageType.LIGHTS)
    agent.enter_vent("v_reactor")
    agent.exit_vent("v_hallway")
    coordinator.update(delta=1.0)

    report = coordinator.status_report()
    score  = agent.behavioral_score(nearby_crew=[(13.0, 9.0), (11.5, 7.8)])
