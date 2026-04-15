"""dataset_exporter.py — Writes complete pedestrian records to CSV."""

import os
from typing import List

import pandas as pd

from pedestrian_gap_analysis.record_store import PedestrianRecord

CSV_COLUMNS = [
    "track_id",
    "gender",
    "age_group",
    "platoon",
    "gap_seconds",
    "time_headway",
    "vehicle_speed",
    "gap_type",
]


class DatasetExporter:
    """Exports complete PedestrianRecord objects to gap_acceptance_dataset.csv."""

    def export(
        self,
        records: List[PedestrianRecord],
        output_dir: str,
        filename: str = "gap_acceptance_dataset.csv",
    ) -> str:
        """
        Write only complete records to CSV.
        Returns the full path to the written file.
        """
        os.makedirs(output_dir, exist_ok=True)

        complete = [r for r in records if r.complete]

        rows = []
        for r in complete:
            rows.append(
                {
                    "track_id": r.track_id,
                    "gender": r.gender,
                    "age_group": r.age_group,
                    "platoon": r.platoon,
                    "gap_seconds": r.gap_seconds,
                    "time_headway": r.time_headway,
                    "vehicle_speed": r.vehicle_speed,
                    # Binary encoding: 1 = Straight, 0 = Rolling
                    "gap_type": r.gap_type_binary,
                }
            )

        df = pd.DataFrame(rows, columns=CSV_COLUMNS)
        out_path = os.path.join(output_dir, filename)
        df.to_csv(out_path, index=False)
        return out_path
