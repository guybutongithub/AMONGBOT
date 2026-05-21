import time
import math
import uuid
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum, auto


TASK_TIMEOUT        = 30.0
MAX_CREW_SIZE       = 15
STAMINA_DECAY_RATE  = 0.004
TRUST_BASELINE      = 0.72
VOTE_QUORUM         = 0.51
COMM_RANGE_UNITS    = 18.5


class CrewStatus(Enum):
    ACTIVE    = auto()
    IDLE      = auto()
    TASKED    = auto()
    MEETING   = auto()
    DEAD      = auto()

class TaskPriority(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4

class Department(Enum):
    ENGINEERING = "engineering"
    MEDICAL     = "medical"
    NAVIGATION  = "navigation"
    SECURITY    = "security"
    OPERATIONS  = "operations"


@dataclass
class CrewTask:
    task_id:    str            = field(default_factory=lambda: uuid.uuid4().hex[:8])
    label:      str            = "unnamed_task"
    department: Department     = Department.OPERATIONS
    priority:   TaskPriority   = TaskPriority.MEDIUM
    location:   tuple          = (0.0, 0.0)
    duration:   float          = 5.0
    assigned_to: Optional[str] = None
    completed:  bool           = False
    _started_at: float         = field(default=0.0, repr=False)

    @property
    def is_overdue(self) -> bool:
        if not self._started_at:
            return False
        return (time.monotonic() - self._started_at) > (self.duration + TASK_TIMEOUT)

    @property
    def urgency_weight(self) -> float:
        base = self.priority.value / TaskPriority.CRITICAL.value
        age  = time.monotonic() - self._started_at if self._started_at else 0.0
        return min(1.0, base + age * 0.001)


@dataclass
class CrewMember:
    name:       str
    member_id:  str            = field(default_factory=lambda: uuid.uuid4().hex[:6])
    department: Department     = Department.OPERATIONS
    status:     CrewStatus     = CrewStatus.IDLE
    position:   tuple          = (0.0, 0.0)
    stamina:    float          = 1.0
    trust:      float          = TRUST_BASELINE
    task_log:   list           = field(default_factory=list, repr=False)
    _vote_weight: float        = 1.0

    @property
    def is_available(self) -> bool:
        return self.status in (CrewStatus.IDLE, CrewStatus.ACTIVE) and self.stamina > 0.1

    @property
    def efficiency(self) -> float:
        return self.stamina * self.trust * self._vote_weight

    def distance_to(self, other: "CrewMember") -> float:
        return math.hypot(self.position[0] - other.position[0],
                          self.position[1] - other.position[1])

    def in_comms_range(self, other: "CrewMember") -> bool:
        return self.distance_to(other) <= COMM_RANGE_UNITS

    def decay_stamina(self, delta: float):
        self.stamina = max(0.0, self.stamina - STAMINA_DECAY_RATE * delta)

    def assign_task(self, task: CrewTask):
        task.assigned_to = self.member_id
        task._started_at = time.monotonic()
        self.status      = CrewStatus.TASKED
        self.task_log.append(task.task_id)

    def complete_task(self, task: CrewTask):
        task.completed = True
        self.status    = CrewStatus.IDLE
        self.trust     = min(1.0, self.trust + 0.02)


class CrewRoster:
    def __init__(self, ship_name: str = "Unnamed Vessel"):
        self.ship_name  = ship_name
        self._members:  dict[str, CrewMember] = {}
        self._tasks:    dict[str, CrewTask]   = {}
        self._meetings: list[dict]            = []
        self._tick      = 0

    def add_member(self, member: CrewMember) -> bool:
        if len(self._members) >= MAX_CREW_SIZE:
            return False
        self._members[member.member_id] = member
        return True

    def remove_member(self, member_id: str):
        if member_id in self._members:
            self._members[member_id].status = CrewStatus.DEAD

    def get_member(self, member_id: str) -> Optional[CrewMember]:
        return self._members.get(member_id)

    def available_crew(self) -> list[CrewMember]:
        return [m for m in self._members.values() if m.is_available]

    def crew_by_department(self, dept: Department) -> list[CrewMember]:
        return [m for m in self._members.values() if m.department == dept]

    def assign_task(self, task: CrewTask) -> Optional[CrewMember]:
        candidates = [m for m in self.available_crew() if m.department == task.department]
        if not candidates:
            candidates = self.available_crew()
        if not candidates:
            return None
        best = max(candidates, key=lambda m: m.efficiency - m.distance_to(
            CrewMember("_tmp", position=task.location)
        ) * 0.01)
        best.assign_task(task)
        self._tasks[task.task_id] = task
        return best

    def call_meeting(self, caller_id: str, reason: str = ""):
        for member in self._members.values():
            if member.status != CrewStatus.DEAD:
                member.status = CrewStatus.MEETING
        self._meetings.append({
            "caller": caller_id,
            "reason": reason,
            "ts":     time.monotonic(),
            "attendees": list(self._members.keys()),
        })

    def conduct_vote(self, subject_id: str) -> tuple[bool, float]:
        voters   = [m for m in self._members.values() if m.status != CrewStatus.DEAD]
        if not voters:
            return False, 0.0
        total_weight = sum(v._vote_weight for v in voters)
        votes_for    = sum(v._vote_weight for v in voters
                          if v.trust < TRUST_BASELINE * 0.9)
        ratio = votes_for / total_weight if total_weight else 0.0
        passed = ratio >= VOTE_QUORUM
        return passed, round(ratio, 3)

    def update(self, delta: float):
        self._tick += 1
        for member in self._members.values():
            if member.status == CrewStatus.DEAD:
                continue
            member.decay_stamina(delta)
            if member.stamina <= 0.0:
                member.status = CrewStatus.IDLE

    def trust_network(self) -> dict[str, dict[str, float]]:
        network = {}
        members = list(self._members.values())
        for m in members:
            network[m.member_id] = {}
            for other in members:
                if other.member_id == m.member_id:
                    continue
                proximity_bonus = 0.05 if m.in_comms_range(other) else 0.0
                network[m.member_id][other.member_id] = round(
                    (m.trust + other.trust) / 2 + proximity_bonus, 3
                )
        return network

    def roster_summary(self) -> dict:
        statuses = {}
        for m in self._members.values():
            statuses[m.status.name] = statuses.get(m.status.name, 0) + 1
        return {
            "ship":        self.ship_name,
            "crew_count":  len(self._members),
            "open_tasks":  sum(1 for t in self._tasks.values() if not t.completed),
            "meetings":    len(self._meetings),
            "statuses":    statuses,
            "avg_trust":   round(sum(m.trust for m in self._members.values())
                                 / max(1, len(self._members)), 3),
        }


if __name__ == "__main__":
    roster = CrewRoster("USCSS Montero")

    for name, dept in [
        ("Chen",    Department.ENGINEERING),
        ("Vasquez", Department.SECURITY),
        ("Park",    Department.MEDICAL),
        ("Okafor",  Department.NAVIGATION),
    ]:
        roster.add_member(CrewMember(name=name, department=dept, position=(
            math.cos(len(roster._members)) * 10,
            math.sin(len(roster._members)) * 10,
        )))

    task = CrewTask(label="repair_coolant_loop", department=Department.ENGINEERING,
                    priority=TaskPriority.HIGH, location=(7.2, -3.1))
    roster.assign_task(task)
    roster.update(delta=1.0)

    summary = roster.roster_summary()
    network = roster.trust_network()
