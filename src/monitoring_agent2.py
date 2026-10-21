import json
import math
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": "192.168.109.154",
    "port": 5433,
    "dbname": "rankpi",
    "user": "rankpi",
    "password": "rankpi",
}

# Polling interval is only an operational scheduling value.
# It is NOT used as a degradation threshold.
MONITOR_INTERVAL_SECONDS = 5


# ============================================================
# DATABASE
# ============================================================

class Database:

    def __init__(self, config):
        self.config = config
        self.conn = None

    def connect(self):
        self.conn = psycopg2.connect(
            host=self.config["host"],
            port=self.config["port"],
            dbname=self.config["dbname"],
            user=self.config["user"],
            password=self.config["password"],
        )
        print("Database connection SUCCESS.")

    def query(self, query_text, params=None):
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query_text, params)

                if cursor.description is None:
                    return []

                columns = [d[0] for d in cursor.description]
                rows = cursor.fetchall()

                return [
                    dict(zip(columns, row))
                    for row in rows
                ]

        except Exception:
            self.conn.rollback()
            raise

    def get_deployed_models(self):
        query_text = """
        WITH deployed_versions AS
        (
            SELECT
                rm.model_id,
                rm.name AS model_name,
                rm.kind AS model_kind,
                rmv.version_id,
                rmv.version,
                rmv.status,
                md.run_id,

                ROW_NUMBER() OVER
                (
                    PARTITION BY rm.model_id
                    ORDER BY rmv.version_id DESC
                ) AS version_rank

            FROM ran_model rm

            JOIN ran_model_version rmv
              ON rmv.model_id = rm.model_id

            LEFT JOIN LATERAL
            (
                SELECT
                    deployment_id,
                    run_id

                FROM model_deployment

                WHERE version_id = rmv.version_id
                  AND ended_at IS NULL

                ORDER BY deployment_id DESC
                LIMIT 1

            ) md ON TRUE

            WHERE rmv.status = 'deployed'
        )

        SELECT
            model_id,
            model_name,
            model_kind,
            version_id,
            version,
            status,
            run_id

        FROM deployed_versions

        WHERE version_rank = 1

        ORDER BY model_id;
        """

        return self.query(query_text)

    def get_model_io(self, version_id):
        query_text = """
        SELECT
            version_id,
            direction,
            ordinal,
            name,
            scope_type,
            source_expr,
            param_id,
            transform,
            scale,
            offset_

        FROM model_io

        WHERE version_id = %s

        ORDER BY direction, ordinal;
        """

        return self.query(query_text, (version_id,))

    def get_view_columns(self, view_name):
        rows = self.query(
            """
            SELECT column_name

            FROM information_schema.columns

            WHERE table_schema = 'public'
              AND table_name = %s

            ORDER BY ordinal_position;
            """,
            (view_name,),
        )

        return {
            row["column_name"]
            for row in rows
        }

    def resolve_model_input_slice(self, input_name):
        """
        Resolve names such as slice1.dl_mbps dynamically from DB.
        """

        if not input_name:
            return None

        token = str(input_name).split(".", 1)[0].strip()

        if not token.lower().startswith("slice"):
            return None

        number = token[5:].strip()

        if not number.isdigit():
            return None

        slice_id = int(number)

        rows = self.query(
            """
            SELECT slice_id

            FROM slice

            WHERE slice_id = %s

            LIMIT 1;
            """,
            (slice_id,),
        )

        return (
            rows[0]["slice_id"]
            if rows
            else None
        )

    def get_latest_source_timestamp(self, view_name):
        columns = self.get_view_columns(view_name)

        if "ts" not in columns:
            return None

        query = sql.SQL(
            """
            SELECT MAX(v.ts) AS latest_ts

            FROM {view} AS v
            """
        ).format(
            view=sql.Identifier(
                "public",
                view_name
            )
        )

        rows = self.query(
            query.as_string(self.conn)
        )

        return (
            rows[0]["latest_ts"]
            if rows
            else None
        )

    def get_source_cadence(self, view_name):
        """
        Learn the source sampling gap from actual timestamps.
        No fixed freshness multiplier is used.
        """

        columns = self.get_view_columns(view_name)

        if "ts" not in columns:
            return None

        query = sql.SQL(
            """
            WITH timestamps AS
            (
                SELECT DISTINCT v.ts

                FROM {view} AS v

                WHERE v.ts IS NOT NULL
            ),

            gaps AS
            (
                SELECT
                    ts - LAG(ts)
                    OVER (ORDER BY ts) AS gap

                FROM timestamps
            )

            SELECT
                COUNT(*) AS n,

                percentile_cont(0.5)
                    WITHIN GROUP
                    (
                        ORDER BY
                        EXTRACT(EPOCH FROM gap)
                    ) AS median_gap,

                percentile_cont(0.75)
                    WITHIN GROUP
                    (
                        ORDER BY
                        EXTRACT(EPOCH FROM gap)
                    ) AS q3_gap,

                MAX(
                    EXTRACT(EPOCH FROM gap)
                ) AS max_gap

            FROM gaps

            WHERE gap IS NOT NULL

              AND EXTRACT(EPOCH FROM gap) > 0
            """
        ).format(
            view=sql.Identifier(
                "public",
                view_name
            )
        )

        rows = self.query(
            query.as_string(self.conn)
        )

        return rows[0] if rows else None

    def get_baseline_stats(
        self,
        view_name,
        column_name,
        scope_id=None,
        run_id=None
    ):
        """
        Learn the baseline distribution inside PostgreSQL.

        Baseline can be restricted to:
        - source view
        - KPI column
        - slice scope
        - deployed model run_id
        """

        available = self.get_view_columns(
            view_name
        )

        if column_name not in available:
            return None

        conditions = [
            sql.SQL(
                "v.{column} IS NOT NULL"
            ).format(
                column=sql.Identifier(
                    column_name
                )
            )
        ]

        params = []

        # Dynamic slice scope
        if scope_id is not None:

            if "slice_id" not in available:
                return None

            conditions.append(
                sql.SQL(
                    "v.slice_id = %s"
                )
            )

            params.append(scope_id)

        # Run-specific baseline
        if (
            run_id is not None
            and "run_id" in available
        ):

            conditions.append(
                sql.SQL(
                    "v.run_id = %s"
                )
            )

            params.append(run_id)

        where_sql = sql.SQL(
            " AND "
        ).join(conditions)

        col = sql.Identifier(
            column_name
        )

        view = sql.Identifier(
            "public",
            view_name
        )

        query = sql.SQL(
            """
            WITH base AS MATERIALIZED
            (
                SELECT
                    v.{col}::double precision AS value

                FROM {view} AS v

                WHERE {where}

                  AND v.{col}::double precision
                      IS NOT NULL
            ),

            central AS
            (
                SELECT
                    COUNT(*) AS n,

                    percentile_cont(0.25)
                        WITHIN GROUP
                        (
                            ORDER BY value
                        ) AS q1,

                    percentile_cont(0.50)
                        WITHIN GROUP
                        (
                            ORDER BY value
                        ) AS median,

                    percentile_cont(0.75)
                        WITHIN GROUP
                        (
                            ORDER BY value
                        ) AS q3

                FROM base
            ),

            absolute_deviations AS
            (
                SELECT
                    ABS(
                        b.value - c.median
                    ) AS deviation

                FROM base b

                CROSS JOIN central c
            ),

            mad_stats AS
            (
                SELECT
                    percentile_cont(0.50)
                        WITHIN GROUP
                        (
                            ORDER BY deviation
                        ) AS mad

                FROM absolute_deviations
            ),

            scale_stats AS
            (
                SELECT
                    GREATEST(
                        1.4826 * COALESCE(
                            m.mad,
                            0.0
                        ),

                        (c.q3 - c.q1) / 1.349
                    ) AS scale

                FROM central c

                CROSS JOIN mad_stats m
            ),

            scored AS
            (
                SELECT
                    ABS(
                        b.value - c.median
                    )
                    /
                    NULLIF(
                        s.scale,
                        0.0
                    ) AS score

                FROM base b

                CROSS JOIN central c

                CROSS JOIN scale_stats s

                WHERE s.scale > 0
            ),

            score_center AS
            (
                SELECT
                    COUNT(*) AS score_n,

                    percentile_cont(0.50)
                        WITHIN GROUP
                        (
                            ORDER BY score
                        ) AS score_median,

                    MAX(score) AS score_max

                FROM scored
            ),

            score_deviations AS
            (
                SELECT
                    ABS(
                        sc.score
                        - scn.score_median
                    ) AS deviation

                FROM scored sc

                CROSS JOIN score_center scn
            ),

            score_mad_stats AS
            (
                SELECT
                    percentile_cont(0.50)
                        WITHIN GROUP
                        (
                            ORDER BY deviation
                        ) AS score_mad

                FROM score_deviations
            ),

            learned_boundary AS
            (
                SELECT
                    sc.score_median,
                    sm.score_mad,
                    sc.score_max,

                    CASE
                        WHEN sm.score_mad > 0

                        THEN
                            sc.score_median
                            +
                            1.4826 * sm.score_mad

                        ELSE
                            sc.score_max
                    END AS score_limit

                FROM score_center sc

                CROSS JOIN score_mad_stats sm
            ),

            score_rate AS
            (
                SELECT
                    CASE
                        WHEN COUNT(*) = 0
                        THEN 0.0

                        ELSE AVG(
                            CASE
                                WHEN
                                    s.score
                                    > b.score_limit

                                THEN 1.0

                                ELSE 0.0
                            END
                        )
                    END
                    AS historical_abnormal_rate

                FROM scored s

                CROSS JOIN learned_boundary b
            )

            SELECT
                c.n,
                c.q1,
                c.median,
                c.q3,

                COALESCE(
                    m.mad,
                    0.0
                ) AS mad,

                COALESCE(
                    s.scale,
                    0.0
                ) AS scale,

                COALESCE(
                    b.score_median,
                    0.0
                ) AS score_median,

                COALESCE(
                    b.score_mad,
                    0.0
                ) AS score_mad,

                COALESCE(
                    b.score_max,
                    0.0
                ) AS score_max,

                COALESCE(
                    b.score_limit,
                    0.0
                ) AS score_limit,

                COALESCE(
                    r.historical_abnormal_rate,
                    0.0
                )
                AS historical_abnormal_rate

            FROM central c

            CROSS JOIN mad_stats m

            CROSS JOIN scale_stats s

            CROSS JOIN learned_boundary b

            CROSS JOIN score_rate r
            """
        ).format(
            col=col,
            view=view,
            where=where_sql,
        )

        rows = self.query(
            query.as_string(
                self.conn
            ),
            tuple(params),
        )

        if (
            not rows
            or not rows[0]["n"]
        ):
            return None

        row = rows[0]

        scale = float(
            row["scale"] or 0.0
        )

        informative = (
            np.isfinite(scale)
            and scale > 0
        )

        return {
            "n": int(row["n"]),

            "median": float(
                row["median"]
            ),

            "q1": float(
                row["q1"]
            ),

            "q3": float(
                row["q3"]
            ),

            "mad": float(
                row["mad"] or 0.0
            ),

            "scale": scale,

            "score_median": float(
                row["score_median"]
                or 0.0
            ),

            "score_mad": float(
                row["score_mad"]
                or 0.0
            ),

            "score_max": float(
                row["score_max"]
                or 0.0
            ),

            "historical_abnormal_rate": float(
                row[
                    "historical_abnormal_rate"
                ]
                or 0.0
            ),

            "informative": bool(
                informative
            ),
        }

    def get_latest_source_values(
        self,
        view_name,
        columns,
        scope_ids,
        run_id=None
    ):
        """
        Fetch the latest required KPI rows.

        IMPORTANT:
        If the source contains run_id and the deployed model has
        a run_id, only KPI data belonging to that same run is used.
        """

        available = self.get_view_columns(
            view_name
        )

        valid_columns = [
            c
            for c in columns
            if c in available
        ]

        if (
            "ts" not in available
            or not valid_columns
        ):
            return pd.DataFrame()

        selected = ["ts"]

        if "slice_id" in available:
            selected.append(
                "slice_id"
            )

        if "run_id" in available:
            selected.append(
                "run_id"
            )

        selected.extend(
            c
            for c in valid_columns
            if c not in selected
        )

        select_sql = sql.SQL(
            ", "
        ).join(
            sql.SQL(
                "v.{c}"
            ).format(
                c=sql.Identifier(c)
            )
            for c in selected
        )

        conditions = []
        params = []

        # ----------------------------------------------------
        # Run-specific current KPI filtering
        # ----------------------------------------------------

        if (
            run_id is not None
            and "run_id" in available
        ):

            conditions.append(
                sql.SQL(
                    "v.run_id = %s"
                )
            )

            params.append(run_id)

        # ----------------------------------------------------
        # Slice filtering
        # ----------------------------------------------------

        if (
            "slice_id" in available
            and scope_ids
        ):

            conditions.append(
                sql.SQL(
                    "v.slice_id = ANY(%s)"
                )
            )

            params.append(
                list(scope_ids)
            )

        view = sql.Identifier(
            "public",
            view_name
        )

        # ----------------------------------------------------
        # Latest row for each discovered slice
        # ----------------------------------------------------

        if (
            "slice_id" in available
            and scope_ids
        ):

            where_sql = (
                sql.SQL("WHERE ")
                +
                sql.SQL(" AND ").join(
                    conditions
                )
            )

            query = sql.SQL(
                """
                SELECT DISTINCT ON
                (
                    v.slice_id
                )
                    {cols}

                FROM {view} AS v

                {where}

                ORDER BY
                    v.slice_id,
                    v.ts DESC
                """
            ).format(
                cols=select_sql,
                view=view,
                where=where_sql,
            )

        else:

            if conditions:

                where_sql = (
                    sql.SQL("WHERE ")
                    +
                    sql.SQL(" AND ").join(
                        conditions
                    )
                )

            else:

                where_sql = sql.SQL("")

            query = sql.SQL(
                """
                SELECT
                    {cols}

                FROM {view} AS v

                {where}

                ORDER BY v.ts DESC

                LIMIT 1
                """
            ).format(
                cols=select_sql,
                view=view,
                where=where_sql,
            )

        rows = self.query(
            query.as_string(
                self.conn
            ),
            tuple(params),
        )

        return pd.DataFrame(rows)

    def get_database_now(self):
        rows = self.query(
            """
            SELECT NOW() AS db_now;
            """
        )

        return (
            rows[0]["db_now"]
            if rows
            else None
        )


# ============================================================
# HELPERS
# ============================================================

def parse_source_expr(source_expr):

    if not source_expr:
        return None, None

    parts = str(
        source_expr
    ).strip().split(".")

    if len(parts) < 2:
        return None, None

    return (
        parts[-2],
        parts[-1]
    )


def safe_float(value):

    try:
        value = float(value)

        return (
            value
            if np.isfinite(value)
            else None
        )

    except (
        TypeError,
        ValueError
    ):
        return None


# ============================================================
# DATA-DRIVEN BASELINE
# ============================================================

class DataDrivenBaseline:

    def __init__(self, stats):

        self.count = int(
            stats["n"]
        )

        self.center = float(
            stats["median"]
        )

        self.q1 = float(
            stats["q1"]
        )

        self.q3 = float(
            stats["q3"]
        )

        self.mad = float(
            stats["mad"]
        )

        self.scale = float(
            stats["scale"]
        )

        self.score_median = float(
            stats["score_median"]
            or 0.0
        )

        self.score_mad = float(
            stats["score_mad"]
            or 0.0
        )

        self.score_max = float(
            stats["score_max"]
            or 0.0
        )

        self.historical_abnormal_rate = float(
            stats.get(
                "historical_abnormal_rate",
                0.0
            )
            or 0.0
        )

        self.informative = bool(
            stats["informative"]
        )

        self.previous_value = None

        self.consecutive_abnormal = 0

    def score(self, value):

        if (
            not self.informative
            or self.scale <= 0
        ):
            return 0.0

        return (
            abs(
                float(value)
                - self.center
            )
            / self.scale
        )

    def learned_score_limit(self):

        """
        Threshold comes from the observed
        baseline score distribution.
        """

        if not self.informative:
            return math.inf

        if self.score_mad > 0:

            return (
                self.score_median
                +
                1.4826
                * self.score_mad
            )

        return self.score_max

    def update_if_normal(self, value):

        """
        Online update using accumulated
        observation count.

        No manually selected learning rate.
        """

        value = safe_float(value)

        if value is None:
            return

        if self.count <= 0:

            self.center = value
            self.count = 1

            return

        old_center = self.center

        self.count += 1

        weight = (
            1.0
            / self.count
        )

        self.center = (
            old_center
            +
            weight
            * (
                value
                - old_center
            )
        )

        deviation = abs(
            value
            - self.center
        )

        self.mad = (
            self.mad
            +
            weight
            * (
                deviation
                - self.mad
            )
        )

        self.scale = max(
            1.4826
            * self.mad,

            (
                self.q3
                - self.q1
            )
            / 1.349
        )


# ============================================================
# BASELINE LEARNER
# ============================================================

class BaselineLearner:

    def __init__(self):

        self.baselines = {}

    def learn(
        self,
        model_id,
        model_name,
        version_id,
        io_rows,
        db
    ):

        print()

        model_run_id = None

        for deployed in db.get_deployed_models():

            if (
                int(
                    deployed["model_id"]
                )
                ==
                int(model_id)
            ):

                model_run_id = (
                    deployed.get(
                        "run_id"
                    )
                )

                break

        print(
            f"Learning data-driven baseline "
            f"for Model {model_id}: "
            f"{model_name}"
        )

        if model_run_id is not None:

            print(
                f"  Baseline scope: "
                f"deployed run_id="
                f"{model_run_id}"
            )

        else:

            print(
                "  Baseline scope: "
                "all available source history "
                "(no active run_id)"
            )

        learned_count = 0

        for row in io_rows:

            if row["direction"] != "input":
                continue

            view_name, column_name = (
                parse_source_expr(
                    row["source_expr"]
                )
            )

            if (
                not view_name
                or not column_name
            ):
                continue

            scope_id = (
                db.resolve_model_input_slice(
                    row["name"]
                )
            )

            stats = db.get_baseline_stats(
                view_name,
                column_name,
                scope_id,
                model_run_id,
            )

            if not stats:
                continue

            key = (
                model_id,
                version_id,
                scope_id,
                view_name,
                column_name,
            )

            self.baselines[key] = (
                DataDrivenBaseline(
                    stats
                )
            )

            learned_count += 1

            state = (
                "informative"
                if stats["informative"]
                else
                "non-informative"
            )

            print(
                f"  {row['name']} | "
                f"scope={scope_id} | "
                f"{view_name}."
                f"{column_name} | "
                f"n={stats['n']} | "
                f"median="
                f"{stats['median']:.6f} | "
                f"MAD="
                f"{stats['mad']:.6f} | "
                f"IQR="
                f"{stats['q1']:.6f}.."
                f"{stats['q3']:.6f} | "
                f"{state}"
            )

        print(
            f"Data-driven baseline signals "
            f"learned: {learned_count}"
        )

        if learned_count:

            print(
                f"Baseline ready for "
                f"Model {model_id}: "
                f"{model_name}"
            )


# ============================================================
# MONITORING AGENT
# ============================================================

class MonitoringAgent:

    def __init__(self, db):

        self.db = db

        self.baseline_learner = (
            BaselineLearner()
        )

        self.initialized_models = set()

        # IMPORTANT:
        # Timestamp is tracked separately for
        # model + version + source + scope + KPI.
        self.last_processed_timestamp = {}

        # Models are initialized from historical data only once.
        # The latest existing KPI timestamp is marked as already
        # processed so historical data is never treated as live data.
        self.initialized_timestamps = set()

        self.last_source_cadence = {}

    def learn_new_model_baselines(
        self,
        models
    ):

        for model in models:

            key = (
                model["model_id"],
                model["version_id"]
            )

            if key in self.initialized_models:
                continue

            print()

            print(
                f"NEW MODEL DISCOVERED: "
                f"{model['model_id']} | "
                f"{model['model_name']}"
            )

            io_rows = (
                self.db.get_model_io(
                    model["version_id"]
                )
            )

            self.baseline_learner.learn(
                model_id=model["model_id"],
                model_name=model["model_name"],
                version_id=model["version_id"],
                io_rows=io_rows,
                db=self.db,
            )

            # Historical data is used only for baseline learning.
            # Mark the latest existing row as already processed so
            # monitoring waits for genuinely new/live data.
            self.initialize_processed_timestamps(
                model,
                io_rows,
            )

            self.initialized_models.add(
                key
            )

    def initialize_processed_timestamps(
        self,
        model,
        io_rows
    ):
        """
        Mark the latest existing historical KPI timestamp for each
        model/version/source/scope/KPI as already processed.

        This prevents the first monitoring cycle from treating the
        latest historical row as newly arrived live data.
        """

        model_id = model["model_id"]
        version_id = model["version_id"]
        model_run_id = model.get("run_id")

        init_key = (
            model_id,
            version_id
        )

        if init_key in self.initialized_timestamps:
            return

        source_groups = self.get_model_sources(
            io_rows
        )

        for view_name, mappings in source_groups.items():

            columns = []
            scope_ids = []

            for mapping in mappings:

                _, column_name = parse_source_expr(
                    mapping["source_expr"]
                )

                if (
                    column_name
                    and column_name not in columns
                ):
                    columns.append(
                        column_name
                    )

                scope_id = (
                    self.resolve_mapping_scope(
                        mapping
                    )
                )

                if (
                    scope_id is not None
                    and scope_id not in scope_ids
                ):
                    scope_ids.append(
                        scope_id
                    )

            if not columns:
                continue

            latest_data = (
                self.db.get_latest_source_values(
                    view_name=view_name,
                    columns=columns,
                    scope_ids=scope_ids,
                    run_id=model_run_id,
                )
            )

            if latest_data.empty:
                continue

            if "ts" not in latest_data.columns:
                continue

            for mapping in mappings:

                _, column_name = parse_source_expr(
                    mapping["source_expr"]
                )

                if (
                    not column_name
                    or column_name not in latest_data.columns
                ):
                    continue

                scope_id = (
                    self.resolve_mapping_scope(
                        mapping
                    )
                )

                temp = latest_data.copy()

                if (
                    scope_id is not None
                    and "slice_id" in temp.columns
                ):
                    temp = temp[
                        temp["slice_id"] == scope_id
                    ]

                if temp.empty:
                    continue

                latest_ts = temp["ts"].max()

                timestamp_key = (
                    model_id,
                    version_id,
                    view_name,
                    scope_id,
                    column_name,
                )

                self.last_processed_timestamp[
                    timestamp_key
                ] = latest_ts

        self.initialized_timestamps.add(
            init_key
        )

    def get_model_sources(
        self,
        io_rows
    ):

        source_groups = defaultdict(
            list
        )

        for row in io_rows:

            if row["direction"] != "input":
                continue

            view_name, column_name = (
                parse_source_expr(
                    row["source_expr"]
                )
            )

            if (
                view_name
                and column_name
            ):

                source_groups[
                    view_name
                ].append(row)

        return source_groups

    def source_is_fresh(
        self,
        view_name,
        latest_ts
    ):

        if latest_ts is None:
            return False, None

        now = (
            self.db.get_database_now()
        )

        if now is None:
            return False, None

        lag = max(
            (
                now
                - latest_ts
            ).total_seconds(),
            0.0
        )

        return True, lag

    def resolve_mapping_scope(
        self,
        mapping
    ):

        return (
            self.db.resolve_model_input_slice(
                mapping["name"]
            )
        )

    def monitor_model(
        self,
        model
    ):

        model_id = model[
            "model_id"
        ]

        model_name = model[
            "model_name"
        ]

        version_id = model[
            "version_id"
        ]

        # IMPORTANT:
        # Get the exact deployed run associated
        # with this model.
        model_run_id = model.get(
            "run_id"
        )

        io_rows = (
            self.db.get_model_io(
                version_id
            )
        )

        source_groups = (
            self.get_model_sources(
                io_rows
            )
        )

        if not source_groups:

            return {
                "model_id": model_id,
                "model_name": model_name,
                "status": "NO_SOURCE_METADATA",
                "degraded": False,
            }

        signal_results = []

        stale_sources = []

        for (
            view_name,
            mappings
        ) in source_groups.items():

            latest_ts = (
                self.db.get_latest_source_timestamp(
                    view_name
                )
            )

            fresh, lag = (
                self.source_is_fresh(
                    view_name,
                    latest_ts
                )
            )

            if not fresh:

                stale_sources.append({
                    "source": view_name,
                    "latest_ts": (
                        str(latest_ts)
                        if latest_ts
                        else None
                    ),
                    "lag_seconds": lag,
                })

                continue

            columns = []

            scope_ids = []

            for mapping in mappings:

                _, column = (
                    parse_source_expr(
                        mapping[
                            "source_expr"
                        ]
                    )
                )

                if (
                    column
                    and column not in columns
                ):

                    columns.append(
                        column
                    )

                scope_id = (
                    self.resolve_mapping_scope(
                        mapping
                    )
                )

                if (
                    scope_id is not None
                    and scope_id not in scope_ids
                ):

                    scope_ids.append(
                        scope_id
                    )

            # IMPORTANT:
            # Current KPI is restricted
            # to the same run_id as the
            # deployed model.
            data = (
                self.db.get_latest_source_values(
                    view_name,
                    columns,
                    scope_ids,
                    model_run_id,
                )
            )

            if (
                data.empty
                or "ts" not in data.columns
            ):
                continue

            # ------------------------------------------------
            # Process every model input independently.
            # ------------------------------------------------

            for mapping in mappings:

                _, column_name = (
                    parse_source_expr(
                        mapping[
                            "source_expr"
                        ]
                    )
                )

                if (
                    not column_name
                    or column_name
                    not in data.columns
                ):
                    continue

                resolved_scope = (
                    self.resolve_mapping_scope(
                        mapping
                    )
                )

                temp = data.copy()

                if (
                    resolved_scope
                    is not None
                    and "slice_id"
                    in temp.columns
                ):

                    temp = temp[
                        temp[
                            "slice_id"
                        ]
                        ==
                        resolved_scope
                    ]

                    scope_id = (
                        resolved_scope
                    )

                else:

                    scope_id = None

                if temp.empty:
                    continue

                # ------------------------------------------------
                # Get the latest row for THIS KPI + THIS SCOPE.
                # ------------------------------------------------

                temp = temp.sort_values(
                    "ts"
                )

                latest_row = (
                    temp.iloc[-1]
                )

                current_timestamp = (
                    latest_row["ts"]
                )

                # ------------------------------------------------
                # NEW-DATA TRACKING
                #
                # Each model + version + source +
                # scope + KPI is tracked independently.
                # ------------------------------------------------

                timestamp_key = (
                    model_id,
                    version_id,
                    view_name,
                    scope_id,
                    column_name,
                )

                previous_ts = (
                    self.last_processed_timestamp.get(
                        timestamp_key
                    )
                )

                # No new data for this specific
                # KPI/scope combination.
                if (
                    previous_ts is not None
                    and current_timestamp
                    <= previous_ts
                ):
                    continue

                # Mark this exact KPI/scope
                # as processed.
                self.last_processed_timestamp[
                    timestamp_key
                ] = current_timestamp

                current_value = safe_float(
                    latest_row[
                        column_name
                    ]
                )

                if current_value is None:
                    continue

                baseline_key = (
                    model_id,
                    version_id,
                    scope_id,
                    view_name,
                    column_name,
                )

                baseline = (
                    self.baseline_learner
                    .baselines
                    .get(
                        baseline_key
                    )
                )

                if baseline is None:
                    continue

                # ------------------------------------------------
                # NON-INFORMATIVE SIGNAL
                # ------------------------------------------------

                if not baseline.informative:

                    signal_results.append({
                        "signal": mapping[
                            "name"
                        ],

                        "source": (
                            f"{view_name}."
                            f"{column_name}"
                        ),

                        "scope_id": scope_id,

                        "value": current_value,

                        "score": 0.0,

                        "learned_limit": None,

                        "abnormal": False,

                        "informative": False,
                    })

                    continue

                # ------------------------------------------------
                # ROBUST ANOMALY SCORE
                # ------------------------------------------------

                score = (
                    baseline.score(
                        current_value
                    )
                )

                limit = (
                    baseline
                    .learned_score_limit()
                )

                abnormal = (
                    score > limit
                )

                # ------------------------------------------------
                # PERSISTENCE TRACKING
                # ------------------------------------------------

                if abnormal:

                    baseline.consecutive_abnormal += 1

                else:

                    baseline.consecutive_abnormal = 0

                    baseline.update_if_normal(
                        current_value
                    )

                signal_results.append({

                    "signal": mapping[
                        "name"
                    ],

                    "source": (
                        f"{view_name}."
                        f"{column_name}"
                    ),

                    "scope_id": scope_id,

                    "value": current_value,

                    "timestamp": (
                        str(
                            current_timestamp
                        )
                    ),

                    "score": score,

                    "learned_limit": limit,

                    "abnormal": abnormal,

                    "informative": True,

                    "consecutive_abnormal":
                        baseline
                        .consecutive_abnormal,

                    "baseline": baseline,
                })

        # ========================================================
        # SOURCE DATA STATUS
        # ========================================================

        if stale_sources:

            return {
                "model_id": model_id,
                "model_name": model_name,
                "status": "STALE_DATA",
                "degraded": False,
                "stale_sources":
                    stale_sources,
                "signals":
                    signal_results,
            }

        if not signal_results:

            return {
                "model_id": model_id,
                "model_name": model_name,
                "status":
                    "WAITING_FOR_NEW_DATA",
                "degraded": False,
            }

        # ========================================================
        # MODEL-LEVEL DECISION
        # ========================================================

        informative = [
            s
            for s in signal_results
            if s["informative"]
        ]

        abnormal = [
            s
            for s in informative
            if s["abnormal"]
        ]

        if not informative:

            model_degraded = False

            status = (
                "NON_INFORMATIVE_SIGNALS"
            )

        else:

            # ------------------------------------------------
            # Historical abnormal rates
            # ------------------------------------------------

            historical_rates = np.array(
                [
                    s["baseline"]
                    .historical_abnormal_rate

                    for s in informative
                ],
                dtype=float,
            )

            rate_center = float(
                np.median(
                    historical_rates
                )
            )

            rate_mad = float(
                np.median(
                    np.abs(
                        historical_rates
                        -
                        rate_center
                    )
                )
            )

            learned_model_rate_limit = (
                rate_center
                +
                1.4826 * rate_mad
                if rate_mad > 0
                else rate_center
            )

            # ------------------------------------------------
            # Persistence analysis
            # ------------------------------------------------

            persistent_abnormal = []

            for signal in abnormal:

                historical_rate = min(
                    max(
                        signal[
                            "baseline"
                        ]
                        .historical_abnormal_rate,
                        0.0,
                    ),
                    1.0,
                )

                expected_run_length = (
                    math.inf
                    if historical_rate >= 1.0

                    else
                    1.0
                    /
                    (
                        1.0
                        -
                        historical_rate
                    )
                )

                if (
                    signal[
                        "consecutive_abnormal"
                    ]
                    >
                    expected_run_length
                ):

                    persistent_abnormal.append(
                        signal
                    )

            persistent_fraction = (
                len(
                    persistent_abnormal
                )
                /
                len(informative)
            )

            # ------------------------------------------------
            # Final model degradation decision
            # ------------------------------------------------

            model_degraded = (
                persistent_fraction
                >
                learned_model_rate_limit

                and

                persistent_fraction
                >
                rate_center
            )

            status = "MONITORING"

        return {

            "model_id": model_id,

            "model_name":
                model_name,

            "status":
                status,

            "degraded":
                bool(
                    model_degraded
                ),

            "abnormal_signal_count":
                len(abnormal),

            "informative_signal_count":
                len(informative),

            "signals":
                signal_results,
        }

    def run_once(self):

        models = (
            self.db.get_deployed_models()
        )

        if not models:

            print(
                "No deployed models found."
            )

            return

        # Discover and learn
        # new models automatically.
        self.learn_new_model_baselines(
            models
        )

        results = []

        for model in models:

            try:

                results.append(
                    self.monitor_model(
                        model
                    )
                )

            except Exception as error:

                results.append({

                    "model_id":
                        model["model_id"],

                    "model_name":
                        model["model_name"],

                    "status":
                        "MONITORING_ERROR",

                    "degraded":
                        False,

                    "error":
                        str(error),
                })

        # ====================================================
        # FINAL JSON OUTPUT
        # ====================================================

        print(
            json.dumps(
                {
                    "monitoring_results":
                    [
                        {
                            "model_id":
                                r[
                                    "model_id"
                                ],

                            "model_name":
                                r[
                                    "model_name"
                                ],

                            "status":
                                r[
                                    "status"
                                ],

                            "degraded":
                                r[
                                    "degraded"
                                ],
                        }

                        for r in results
                    ]
                },

                indent=2,

                default=str
            )
        )

        # ====================================================
        # ABNORMAL SIGNAL EVIDENCE
        # ====================================================

        for result in results:

            abnormal = [

                s

                for s in result.get(
                    "signals",
                    []
                )

                if s.get(
                    "abnormal"
                )
            ]

            if abnormal:

                print(
                    f"\nEvidence for Model "
                    f"{result['model_id']} "
                    f"({result['model_name']}):"
                )

                for signal in abnormal:

                    print(

                        f"  "
                        f"{signal['signal']} | "

                        f"scope="
                        f"{signal['scope_id']} | "

                        f"score="
                        f"{signal['score']:.4f} | "

                        f"learned_limit="
                        f"{signal['learned_limit']:.4f}"
                    )

    def run_forever(self):

        print()

        print(
            "Generic AI Monitoring Agent started."
        )

        print(
            f"Polling interval: "
            f"{MONITOR_INTERVAL_SECONDS}s"
        )

        print(
            "Model IDs, model inputs, "
            "source views, source columns, "
            "slice scopes, baselines and "
            "decision boundaries are "
            "discovered/learned from the "
            "database."
        )

        print()

        while True:

            try:

                self.run_once()

            except Exception as error:

                print()

                print(
                    "Monitoring error:"
                )

                print(error)

            time.sleep(
                MONITOR_INTERVAL_SECONDS
            )


# ============================================================
# MAIN
# ============================================================

def main():

    db = Database(
        DB_CONFIG
    )

    try:

        db.connect()

        agent = MonitoringAgent(
            db
        )

        agent.run_forever()

    except KeyboardInterrupt:

        print(
            "\nMonitoring stopped."
        )

    except Exception as error:

        print(
            f"Startup error: "
            f"{error}"
        )

    finally:

        if db.conn is not None:
            db.conn.close()


if __name__ == "__main__":
    main()