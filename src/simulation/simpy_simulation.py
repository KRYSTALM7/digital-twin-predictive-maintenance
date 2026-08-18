import os
import csv
import math
import random
from enum import Enum
from typing import List, Tuple

import simpy


class SpindleState(Enum):
    RUNNING = "RUNNING"
    IDLE = "IDLE"
    FAILURE = "FAILURE"
    MAINTENANCE = "MAINTENANCE"


class CncSpindleSimulation:
    """Simulates a CNC spindle with failures and preventive maintenance.

    At each environment step, emits synthetic sensor data.
    """

    def __init__(
        self,
        env: simpy.Environment,
        mean_time_to_failure: float = 300.0,
        mean_repair_time: float = 50.0,
        preventive_maintenance_interval: float = 200.0,
        timestep: float = 1.0,
    ) -> None:
        self.env = env
        self.mean_time_to_failure = mean_time_to_failure
        self.mean_repair_time = mean_repair_time
        self.preventive_maintenance_interval = preventive_maintenance_interval
        self.timestep = timestep

        self.state = SpindleState.RUNNING
        self.next_failure_time = self._sample_next_failure_time()
        self.next_pm_time = preventive_maintenance_interval

        self.wear_level = 0.0
        self.wear_drift_per_time = 0.003

        # Now includes a ground-truth failure flag (1 during FAILURE state)
        self.records: List[Tuple[float, str, float, float, float, int]] = []

    def _sample_next_failure_time(self) -> float:
        rate = 1.0 / self.mean_time_to_failure
        return self.env.now + random.expovariate(rate)

    def _sample_repair_duration(self) -> float:
        rate = 1.0 / self.mean_repair_time
        return random.expovariate(rate)

    def _generate_sensors(self, state: SpindleState) -> Tuple[float, float, float]:
        temperature = random.gauss(70.0, 2.0)
        vibration = random.gauss(0.02, 0.005)
        current_baseline = 10.0 + self.wear_level
        current = random.gauss(current_baseline, 0.5)

        if state == SpindleState.FAILURE:
            temperature += 10.0
            vibration *= 3.0
        elif state in (SpindleState.IDLE, SpindleState.MAINTENANCE):
            temperature -= 3.0
            vibration *= 0.5
            current -= 2.0

        return temperature, max(vibration, 0.0), max(current, 0.0)

    def _log_step(self) -> None:
        temperature, vibration, current = self._generate_sensors(self.state)
        ground_truth_failure = 1 if self.state == SpindleState.FAILURE else 0
        self.records.append(
            (
                self.env.now,
                self.state.value,
                float(temperature),
                float(vibration),
                float(current),
                ground_truth_failure,
            )
        )

    def _apply_wear(self) -> None:
        if self.state == SpindleState.RUNNING:
            self.wear_level += self.wear_drift_per_time * self.timestep

    def _run_for_duration(self, duration: float):
        """Step through `duration` in timestep increments, logging every step.
        This is what makes FAILURE/MAINTENANCE periods show up in the CSV
        instead of being skipped over as a single yield."""
        elapsed = 0.0
        while elapsed < duration:
            step = min(self.timestep, duration - elapsed)
            self._log_step()
            yield self.env.timeout(step)
            elapsed += step

    def process(self):  # type: ignore[override]
        while True:
            time_to_failure = max(self.next_failure_time - self.env.now, 0.0)
            time_to_pm = max(self.next_pm_time - self.env.now, 0.0)

            if (
                self.state == SpindleState.RUNNING
                and time_to_pm <= self.timestep
                and time_to_pm <= time_to_failure
            ):
                self.state = SpindleState.MAINTENANCE
                pm_duration = max(10.0, 0.3 * self.mean_repair_time)
                yield from self._run_for_duration(pm_duration)
                self.wear_level = 0.0
                self.next_pm_time = self.env.now + self.preventive_maintenance_interval
                self.state = SpindleState.RUNNING
                self.next_failure_time = self._sample_next_failure_time()
                continue

            if self.state == SpindleState.RUNNING and time_to_failure <= self.timestep:
                self.state = SpindleState.FAILURE
                repair_duration = self._sample_repair_duration()
                yield from self._run_for_duration(repair_duration)
                self.wear_level = 0.0
                self.next_failure_time = self._sample_next_failure_time()
                self.state = SpindleState.RUNNING
                continue

            self._apply_wear()
            self._log_step()
            yield self.env.timeout(self.timestep)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_csv(path: str, rows: List[Tuple[float, str, float, float, float, int]]) -> None:
    header = ["timestamp", "state", "temperature", "vibration", "current", "ground_truth_failure"]
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def run_simulation(
    duration: float = 1000.0,
    mean_time_to_failure: float = 300.0,
    mean_repair_time: float = 50.0,
    preventive_maintenance_interval: float = 200.0,
    output_csv_path: str = os.path.join("data", "raw", "simulated_spindle_data.csv"),
) -> str:
    env = simpy.Environment()
    spindle = CncSpindleSimulation(
        env=env,
        mean_time_to_failure=mean_time_to_failure,
        mean_repair_time=mean_repair_time,
        preventive_maintenance_interval=preventive_maintenance_interval,
        timestep=1.0,
    )
    env.process(spindle.process())
    env.run(until=duration)

    out_dir = os.path.dirname(output_csv_path)
    if out_dir:
        _ensure_dir(out_dir)
    _write_csv(output_csv_path, spindle.records)
    return output_csv_path


if __name__ == "__main__":
    path = run_simulation()
    print(f"Simulation complete. CSV saved to: {path}")