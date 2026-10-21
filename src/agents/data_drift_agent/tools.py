import psycopg2


# --------------------------------------------------
# DATABASE CONFIGURATION
# --------------------------------------------------

DB_CONFIG = {
    "host": "192.168.109.154",
    "port": 5433,
    "database": "rankpi",
    "user": "rankpi",
    "password": "rankpi"
}


# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------

def connect_to_database():
    """Connect to the PostgreSQL database."""
    return psycopg2.connect(**DB_CONFIG)


# --------------------------------------------------
# TOOL 1: GET DEPLOYED MODEL
# --------------------------------------------------

def get_deployed_model():
    """
    Fetch the currently deployed model and its version
    from the PostgreSQL database.
    """

    conn = connect_to_database()

    try:
        query = """
            SELECT
                md.deployment_id,
                md.run_id,
                rmv.version_id,
                rmv.model_id,
                rm.name AS model_name,
                rm.kind,
                rm.objective,
                rmv.version,
                rmv.algorithm,
                rmv.status
            FROM model_deployment md
            JOIN ran_model_version rmv
                ON md.version_id = rmv.version_id
            JOIN ran_model rm
                ON rmv.model_id = rm.model_id
            WHERE md.deployment_id = (
                SELECT MAX(deployment_id)
                FROM model_deployment
            );
        """

        with conn.cursor() as cursor:
            cursor.execute(query)
            row = cursor.fetchone()

        if row is None:
            return None

        return {
            "deployment_id": row[0],
            "run_id": row[1],
            "version_id": row[2],
            "model_id": row[3],
            "model_name": row[4],
            "kind": row[5],
            "objective": row[6],
            "version": row[7],
            "algorithm": row[8],
            "status": row[9]
        }

    finally:
        conn.close()


# --------------------------------------------------
# TOOL 2: GET MODEL INPUTS
# --------------------------------------------------

def get_model_inputs(version_id):
    """
    Fetch all input features used by a model version.
    """

    conn = connect_to_database()

    try:
        query = """
            SELECT
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
              AND direction = 'input'
            ORDER BY ordinal;
        """

        with conn.cursor() as cursor:
            cursor.execute(query, (version_id,))
            rows = cursor.fetchall()

        features = []

        for row in rows:
            features.append({
                "ordinal": row[0],
                "name": row[1],
                "scope_type": row[2],
                "source_expr": row[3],
                "param_id": row[4],
                "transform": row[5],
                "scale": row[6],
                "offset": row[7]
            })

        return features

    finally:
        conn.close()


# --------------------------------------------------
# TOOL 3: GET FEATURE DATA
# --------------------------------------------------

def get_feature_data(run_id, slice_id, source_column, limit=10):
    """
    Fetch KPI data for a specific model input feature
    from the v_ue_rate view.

    The query intentionally does not use ORDER BY for the
    initial connectivity/performance test.
    """

    # Only allow known KPI columns.
    # This prevents an arbitrary SQL identifier from being used.
    allowed_columns = {
        "dl_mbps",
        "ul_mbps",
        "d_dl_prb",
        "d_ul_prb",
        "dl_prb_util",
        "ul_prb_util",
        "pusch_snr",
        "pucch_snr",
        "wb_cqi",
        "dl_mcs1",
        "ul_mcs1",
        "dl_bler",
        "ul_bler",
        "phr",
        "bsr"
    }

    if source_column not in allowed_columns:
        raise ValueError(
            f"Unsupported KPI column: {source_column}"
        )

    conn = connect_to_database()

    try:
        query = f"""
            SELECT
                ts,
                run_id,
                cell_id,
                rnti,
                session_id,
                ue_id,
                slice_id,
                {source_column}
            FROM v_ue_rate
            WHERE run_id = %s
              AND slice_id = %s
              AND {source_column} IS NOT NULL
            LIMIT %s;
        """

        with conn.cursor() as cursor:

            cursor.execute(
                query,
                (
                    run_id,
                    slice_id,
                    limit
                )
            )

            rows = cursor.fetchall()

        return [
            {
                "ts": row[0],
                "run_id": row[1],
                "cell_id": row[2],
                "rnti": row[3],
                "session_id": row[4],
                "ue_id": row[5],
                "slice_id": row[6],
                "value": row[7]
            }
            for row in rows
        ]

    finally:
        conn.close()