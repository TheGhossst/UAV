"""Discrete-event AoDT queues: FCFS, FCFS-P, and paper LCFS-S.

Paper §III: FCFS leaves stale packets in line; LCFS with preemption in
service (LCFS-S) is the discipline used to motivate Eqs. (14)–(17).
Problem (P) still scores the closed form (17), not this simulator.

Disciplines
-----------
``lcfs_s``
    Newest packet preempts service and all waiting packets are dropped.
    Equivalent to M/M/1/1 replacement. Paper's named queue.

``fcfs``
    Infinite buffer, no preemption. Stale packets are served.

``fcfs_p``
    FCFS service (non-preemptive) with same-source waiting-room
    replacement: a new packet from i drops any *queued* packet from i.
    In-service work is not preempted. This is packet-management FCFS,
    not the paper's LCFS-S.

Each process is queued on its processing UAV in isolation (Eq. (17)
uses only Σ_{i∈N_k} λ_i). Set ``queue_scope='uav'`` to share the
server across processes assigned to the same UAV (constraint (24)
load), which is *not* how (17) is written.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import heapq

import numpy as np

from uavdt.aodt import download_time_s, upload_times_s
from uavdt.computation import sample_interarrival_times_s, service_rate_per_s
from uavdt.models import Scenario


class Discipline(str, Enum):
    FCFS = "fcfs"
    FCFS_P = "fcfs_p"
    LCFS_S = "lcfs_s"


DISCIPLINES: tuple[str, ...] = tuple(d.value for d in Discipline)


@dataclass
class _Packet:
    source: int
    t_gen: float
    t_arrive: float


@dataclass
class SourceSimResult:
    mean_age_s: float
    n_generated: int
    n_delivered: int
    n_dropped: int


@dataclass
class ProcessSimResult:
    process_id: int
    discipline: str
    mean_process_age_s: float
    mean_source_age_s: np.ndarray
    sources: tuple[SourceSimResult, ...]
    n_generated: int
    n_delivered: int
    n_dropped: int
    horizon_s: float
    warmup_s: float
    eq10_t_s: np.ndarray | None = None
    eq10_age_s: np.ndarray | None = None  # (n_samples, n_sources)


@dataclass
class ScenarioSimResult:
    discipline: str
    queue_scope: str
    process: tuple[ProcessSimResult, ...]
    mean_process_age_s: np.ndarray
    mean_max_process_age_s: float


@dataclass
class _AgeClock:
    """Trapezoidal integral of Eq. (10) ages, with warmup."""

    n: int
    warmup_s: float
    horizon_s: float
    u: np.ndarray
    t: float = 0.0
    integral: np.ndarray = field(init=False)
    integral_max: float = 0.0
    measure_s: float = 0.0
    sample_t: list[float] = field(default_factory=list)
    sample_age: list[np.ndarray] = field(default_factory=list)
    record_eq10: bool = False

    def __post_init__(self) -> None:
        self.u = np.zeros(self.n, dtype=float)
        self.integral = np.zeros(self.n, dtype=float)

    def advance(self, t_new: float) -> None:
        t_new = float(t_new)
        if t_new <= self.t + 1e-18:
            return
        t0, t1 = self.t, t_new
        ages0 = t0 - self.u
        ages1 = t1 - self.u
        lo = max(t0, self.warmup_s)
        hi = min(t1, self.horizon_s)
        if hi > lo:
            # Linear ages: interpolate the clipped window.
            a_lo = ages0 + (lo - t0)
            a_hi = ages0 + (hi - t0)
            dt = hi - lo
            self.integral += 0.5 * dt * (a_lo + a_hi)
            self.integral_max += 0.5 * dt * (float(np.max(a_lo)) + float(np.max(a_hi)))
            self.measure_s += dt
        self.t = t1
        if self.record_eq10 and t_new <= self.horizon_s + 1e-15:
            self.sample_t.append(t_new)
            self.sample_age.append(t_new - self.u.copy())

    def deliver(self, source: int, t_gen: float, t_now: float) -> None:
        self.advance(t_now)
        self.u[int(source)] = float(t_gen)
        if self.record_eq10:
            self.sample_t.append(t_now)
            self.sample_age.append(t_now - self.u.copy())

    def finish(self) -> None:
        self.advance(self.horizon_s)

    def mean_ages(self) -> np.ndarray:
        if self.measure_s <= 0.0:
            return np.full(self.n, np.inf)
        return self.integral / self.measure_s

    def mean_max_age(self) -> float:
        if self.measure_s <= 0.0:
            return np.inf
        return self.integral_max / self.measure_s


def _arrivals(
    lambdas_per_s: np.ndarray,
    delays_s: np.ndarray,
    horizon_s: float,
    rng: np.random.Generator,
) -> list[tuple[float, int, float]]:
    """(t_arrive, source, t_gen) with t_arrive <= horizon."""
    out: list[tuple[float, int, float]] = []
    for i, (lam, delay) in enumerate(zip(lambdas_per_s, delays_s)):
        gens = sample_interarrival_times_s(float(lam), horizon_s, rng)
        d = float(delay)
        for tg in gens:
            ta = float(tg) + d
            if ta <= horizon_s:
                out.append((ta, i, float(tg)))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def simulate_process_queue(
    lambdas_per_s: np.ndarray,
    delays_s: np.ndarray,
    mu_per_s: float,
    *,
    discipline: str = Discipline.LCFS_S.value,
    horizon_s: float = 200.0,
    warmup_s: float = 40.0,
    download_s: float = 0.0,
    rng: np.random.Generator | None = None,
    seed: int = 0,
    record_eq10: bool = False,
    process_id: int = 0,
) -> ProcessSimResult:
    """Simulate one process's UAV queue (isolated N_k traffic)."""
    name = Discipline(discipline).value
    if rng is None:
        rng = np.random.default_rng(seed)
    lam = np.asarray(lambdas_per_s, dtype=float)
    delay = np.asarray(delays_s, dtype=float)
    n = int(lam.size)
    if n == 0:
        return ProcessSimResult(
            process_id, name, np.inf, np.zeros(0), tuple(), 0, 0, 0,
            horizon_s, warmup_s,
        )
    if not np.isfinite(mu_per_s) or mu_per_s <= 0.0:
        inf_src = tuple(SourceSimResult(np.inf, 0, 0, 0) for _ in range(n))
        return ProcessSimResult(
            process_id, name, np.inf, np.full(n, np.inf), inf_src, 0, 0, 0,
            horizon_s, warmup_s,
        )

    arrivals = _arrivals(lam, delay, horizon_s, rng)
    n_gen = np.zeros(n, dtype=int)
    for _, src, _ in arrivals:
        n_gen[src] += 1

    clock = _AgeClock(
        n=n, warmup_s=warmup_s, horizon_s=horizon_s, u=np.zeros(n),
        record_eq10=record_eq10,
    )
    n_del = np.zeros(n, dtype=int)
    n_drop = np.zeros(n, dtype=int)

    if name == Discipline.LCFS_S.value:
        _run_lcfs_s(
            arrivals, mu_per_s, download_s, horizon_s, rng, clock, n_del, n_drop
        )
    else:
        replace_waiting = name == Discipline.FCFS_P.value
        _run_fcfs(
            arrivals, mu_per_s, download_s, horizon_s, rng, clock,
            n_del, n_drop, replace_waiting=replace_waiting,
        )

    clock.finish()
    means = clock.mean_ages()
    sources = tuple(
        SourceSimResult(
            mean_age_s=float(means[i]),
            n_generated=int(n_gen[i]),
            n_delivered=int(n_del[i]),
            n_dropped=int(n_drop[i]),
        )
        for i in range(n)
    )
    eq10_t = eq10_a = None
    if record_eq10 and clock.sample_t:
        eq10_t = np.asarray(clock.sample_t, dtype=float)
        eq10_a = np.vstack(clock.sample_age)
    return ProcessSimResult(
        process_id=process_id,
        discipline=name,
        mean_process_age_s=float(clock.mean_max_age()),
        mean_source_age_s=means,
        sources=sources,
        n_generated=int(n_gen.sum()),
        n_delivered=int(n_del.sum()),
        n_dropped=int(n_drop.sum()),
        horizon_s=horizon_s,
        warmup_s=warmup_s,
        eq10_t_s=eq10_t,
        eq10_age_s=eq10_a,
    )


def _exp_service(mu: float, rng: np.random.Generator) -> float:
    return float(rng.exponential(1.0 / mu))


def _run_lcfs_s(
    arrivals: list[tuple[float, int, float]],
    mu: float,
    download_s: float,
    horizon_s: float,
    rng: np.random.Generator,
    clock: _AgeClock,
    n_del: np.ndarray,
    n_drop: np.ndarray,
) -> None:
    """Newest packet only; preempt-and-delete (paper LCFS-S)."""
    in_srv: _Packet | None = None
    t_done = np.inf
    for ta, src, tg in arrivals:
        if ta > horizon_s:
            break
        if in_srv is not None and t_done <= ta:
            _complete(in_srv, t_done, download_s, horizon_s, clock, n_del)
            in_srv = None
            t_done = np.inf
        if in_srv is not None:
            n_drop[in_srv.source] += 1
            in_srv = None
        in_srv = _Packet(src, tg, ta)
        clock.advance(ta)
        t_done = ta + _exp_service(mu, rng)
    if in_srv is not None and t_done <= horizon_s:
        _complete(in_srv, t_done, download_s, horizon_s, clock, n_del)


def _run_fcfs(
    arrivals: list[tuple[float, int, float]],
    mu: float,
    download_s: float,
    horizon_s: float,
    rng: np.random.Generator,
    clock: _AgeClock,
    n_del: np.ndarray,
    n_drop: np.ndarray,
    *,
    replace_waiting: bool,
) -> None:
    queue: deque[_Packet] = deque()
    in_srv: _Packet | None = None
    t_done = np.inf
    events: list[tuple[float, int, int]] = []
    # (time, kind, idx) kind 0=depart token, 1=arrival index
    # Departures are scheduled dynamically; start with arrivals only.
    for i, (ta, _, _) in enumerate(arrivals):
        heapq.heappush(events, (ta, 1, i))
    depart_seq = 0
    valid_depart = -1

    def start_service(pkt: _Packet, t_now: float) -> None:
        nonlocal in_srv, t_done, depart_seq, valid_depart
        in_srv = pkt
        t_done = t_now + _exp_service(mu, rng)
        depart_seq += 1
        valid_depart = depart_seq
        heapq.heappush(events, (t_done, 0, depart_seq))

    while events:
        t, kind, token = heapq.heappop(events)
        if t > horizon_s:
            break
        if kind == 0:
            if token != valid_depart or in_srv is None:
                continue
            _complete(in_srv, t, download_s, horizon_s, clock, n_del)
            in_srv = None
            valid_depart = -1
            t_done = np.inf
            if queue:
                start_service(queue.popleft(), t)
            continue
        ta, src, tg = arrivals[token]
        clock.advance(ta)
        pkt = _Packet(src, tg, ta)
        if replace_waiting:
            kept: deque[_Packet] = deque()
            while queue:
                q = queue.popleft()
                if q.source == src:
                    n_drop[q.source] += 1
                else:
                    kept.append(q)
            queue = kept
        if in_srv is None:
            start_service(pkt, ta)
        else:
            queue.append(pkt)


def _complete(
    pkt: _Packet,
    t_depart: float,
    download_s: float,
    horizon_s: float,
    clock: _AgeClock,
    n_del: np.ndarray,
) -> None:
    t_rx = t_depart + download_s
    if t_rx > horizon_s:
        return
    clock.deliver(pkt.source, pkt.t_gen, t_rx)
    n_del[pkt.source] += 1


def simulate_scenario_aodt(
    scenario: Scenario,
    association: np.ndarray,
    processing: np.ndarray,
    rates_bit_per_s: np.ndarray,
    *,
    discipline: str = Discipline.LCFS_S.value,
    queue_scope: str = "process",
    horizon_s: float = 200.0,
    warmup_s: float = 40.0,
    rng: np.random.Generator | None = None,
    seed: int = 0,
    record_eq10: bool = False,
) -> ScenarioSimResult:
    """Simulate AoDT for every process under one queueing discipline."""
    if rng is None:
        rng = np.random.default_rng(seed)
    cfg = scenario.cfg
    d_i = upload_times_s(association, processing, rates_bit_per_s, cfg)
    z = download_time_s(cfg)
    mu = float(service_rate_per_s(cfg))
    name = Discipline(discipline).value
    scope = queue_scope.strip().lower()
    results: list[ProcessSimResult] = []

    if scope == "process":
        for proc in scenario.processes:
            members = proc.iot_indices
            child = np.random.default_rng(rng.integers(0, 2**31 - 1))
            results.append(
                simulate_process_queue(
                    scenario.lambdas_per_s[members],
                    d_i[members],
                    mu,
                    discipline=name,
                    horizon_s=horizon_s,
                    warmup_s=warmup_s,
                    download_s=z,
                    rng=child,
                    record_eq10=record_eq10,
                    process_id=proc.process_id,
                )
            )
    elif scope == "uav":
        j_count = int(processing.shape[1])
        uav_members: list[list[int]] = [[] for _ in range(j_count)]
        uav_proc: list[list[int]] = [[] for _ in range(j_count)]
        for proc in scenario.processes:
            members = proc.iot_indices
            if members.size == 0:
                continue
            j_star = int(np.argmax(processing[members[0]]))
            uav_members[j_star].extend(int(i) for i in members)
            uav_proc[j_star].append(proc.process_id)
        by_proc: dict[int, ProcessSimResult] = {}
        for j in range(j_count):
            idxs = np.asarray(uav_members[j], dtype=int)
            if idxs.size == 0:
                continue
            child = np.random.default_rng(rng.integers(0, 2**31 - 1))
            shared = simulate_process_queue(
                scenario.lambdas_per_s[idxs],
                d_i[idxs],
                mu,
                discipline=name,
                horizon_s=horizon_s,
                warmup_s=warmup_s,
                download_s=z,
                rng=child,
                record_eq10=record_eq10,
                process_id=-j,
            )
            offset = 0
            for proc in scenario.processes:
                if proc.process_id not in uav_proc[j]:
                    continue
                m = proc.iot_indices.size
                src_slice = shared.mean_source_age_s[offset : offset + m]
                src_stats = shared.sources[offset : offset + m]
                by_proc[proc.process_id] = ProcessSimResult(
                    process_id=proc.process_id,
                    discipline=name,
                    mean_process_age_s=float(np.max(src_slice))
                    if src_slice.size
                    else np.inf,
                    mean_source_age_s=src_slice,
                    sources=src_stats,
                    n_generated=sum(s.n_generated for s in src_stats),
                    n_delivered=sum(s.n_delivered for s in src_stats),
                    n_dropped=sum(s.n_dropped for s in src_stats),
                    horizon_s=horizon_s,
                    warmup_s=warmup_s,
                )
                offset += m
        for proc in scenario.processes:
            if proc.process_id in by_proc:
                results.append(by_proc[proc.process_id])
            else:
                results.append(
                    ProcessSimResult(
                        proc.process_id, name, np.inf, np.zeros(0), tuple(),
                        0, 0, 0, horizon_s, warmup_s,
                    )
                )
    else:
        raise ValueError("queue_scope must be 'process' or 'uav'")

    ages = np.array([r.mean_process_age_s for r in results], dtype=float)
    return ScenarioSimResult(
        discipline=name,
        queue_scope=scope,
        process=tuple(results),
        mean_process_age_s=ages,
        mean_max_process_age_s=float(np.max(ages)) if ages.size else np.inf,
    )


def compare_closed_and_sim(
    scenario: Scenario,
    association: np.ndarray,
    processing: np.ndarray,
    rates_bit_per_s: np.ndarray,
    mu_per_uav: np.ndarray,
    uav_stable: np.ndarray | None = None,
    *,
    disciplines: tuple[str, ...] = DISCIPLINES,
    horizon_s: float = 120.0,
    warmup_s: float = 24.0,
    seed: int = 0,
    queue_scope: str = "process",
) -> dict:
    """Closed forms (14)–(17)/FCFS vs event-driven queues for one deployment."""
    from uavdt.aodt import (
        average_aodt_eq15_s,
        average_aodt_fcfs_closed_s,
        average_aodt_s,
    )

    eq17 = average_aodt_s(
        scenario, association, processing, rates_bit_per_s, mu_per_uav, uav_stable
    )
    eq15 = average_aodt_eq15_s(
        scenario, association, processing, rates_bit_per_s, mu_per_uav, uav_stable
    )
    fcfs_c = average_aodt_fcfs_closed_s(
        scenario, association, processing, rates_bit_per_s, mu_per_uav, uav_stable
    )
    sims = {}
    for d in disciplines:
        sim = simulate_scenario_aodt(
            scenario,
            association,
            processing,
            rates_bit_per_s,
            discipline=d,
            queue_scope=queue_scope,
            horizon_s=horizon_s,
            warmup_s=warmup_s,
            seed=seed,
        )
        src_rows = [p.mean_source_age_s for p in sim.process]
        flat = [float(x) for row in src_rows for x in row]
        sims[d] = {
            "mean_process_age_s": [float(x) for x in sim.mean_process_age_s],
            "mean_max_process_age_s": float(sim.mean_max_process_age_s),
            "mean_source_age_s": [
                [float(x) for x in p.mean_source_age_s] for p in sim.process
            ],
            "mean_source_age_flat_s": float(np.mean(flat)) if flat else np.inf,
            "n_delivered": [int(p.n_delivered) for p in sim.process],
            "n_dropped": [int(p.n_dropped) for p in sim.process],
        }
    return {
        "eq17_s": [float(x) for x in eq17],
        "eq15_s": [float(x) for x in eq15],
        "fcfs_closed_s": [float(x) for x in fcfs_c],
        "eq17_max_s": float(np.max(eq17)),
        "eq15_max_s": float(np.max(eq15)),
        "fcfs_closed_max_s": float(np.max(fcfs_c)),
        "sim": sims,
        "horizon_s": horizon_s,
        "warmup_s": warmup_s,
        "queue_scope": queue_scope,
    }

