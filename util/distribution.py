from pyspark.sql import functions as F
from pyspark.sql import Window

def distribute_largest_remainder(
    df,
    total_col,
    pct_col,
    group_cols=None,
    id_col=None,
    result_col="assigned_total"
):
    """
    Distribute integer quantities using the Largest Remainder Method
    with support for multiple grouping columns.

    Parameters
    ----------
    df : DataFrame
        Input data.
    total_col : str
        Column with total units to assign (can vary per group).
    pct_col : str
        Column with fractional percentage (0–1).
    group_cols : list[str] or None
        Columns to group by. If None → global distribution.
    id_col : str or None
        Tie-break column. If None, a unique id will be generated.
    result_col : str
        Output column with assigned integer units.

    Returns
    -------
    DataFrame
    """
    # 1) Compute ideal, integer part, remainder
    df1 = (
        df.withColumn("ideal", F.col(pct_col) * F.col(total_col))
          .withColumn("integer", F.floor("ideal"))
          .withColumn("remainder", F.col("ideal") - F.col("integer"))
    )

    # 2) Generate tie-break ID if needed
    if id_col is None:
        df1 = df1.withColumn("_tie_id", F.monotonically_increasing_id())
        tie_col = "_tie_id"
    else:
        tie_col = id_col

    # ---- CASE 1: No grouping → global distribution ----
    if group_cols is None or len(group_cols) == 0:
        missing = (
            df1.agg((F.first(total_col) - F.sum("integer")).alias("missing_units"))
                .collect()[0]["missing_units"]
        )
        missing_units = int(missing) if missing is not None else 0

        if missing_units <= 0:
            return df1.withColumn(result_col, F.col("integer")).drop("_tie_id", "ideal", "remainder")

        w = Window.orderBy(F.col("remainder").desc(), F.col(tie_col))

        df2 = (
            df1.withColumn("rn", F.row_number().over(w))
                .withColumn("extra", F.when(F.col("rn") <= missing_units, 1).otherwise(0))
                .withColumn(result_col, F.col("integer") + F.col("extra"))
        )

        return df2.drop("ideal", "integer", "remainder", "rn", "extra", "_tie_id")

    # ---- CASE 2: Grouped distribution with multiple columns ----

    # Compute missing_units per group
    grp = (
        df1.groupBy(group_cols)
           .agg(
               F.first(total_col).alias("_group_total"),
               F.sum("integer").alias("_sum_integer")
           )
           .withColumn("missing_units", (F.col("_group_total") - F.col("_sum_integer")).cast("int"))
           .select(*group_cols, "missing_units")
    )

    df_join = df1.join(grp, on=group_cols, how="left")

    # Window partitioned by all group cols
    w = Window.partitionBy(*group_cols).orderBy(
        F.col("remainder").desc(),
        F.col(tie_col)
    )

    df2 = (
        df_join.withColumn("rn", F.row_number().over(w))
               .withColumn(
                   "extra",
                   F.when(
                       (F.col("missing_units") > 0) & (F.col("rn") <= F.col("missing_units")),
                       1
                   ).otherwise(0)
               )
               .withColumn(result_col, F.col("integer") + F.col("extra"))
    )

    return df2.drop(
        "ideal", "integer", "remainder", "rn", "extra",
        "_group_total", "_sum_integer", "missing_units", "_tie_id"
    )
