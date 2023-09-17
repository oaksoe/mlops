#!/bin/sh

sqlite3 mlflow-experiment-tracking.db <<'END_SQL'
.output ./database-state.sql
.dump
END_SQL

rm -rf mlflow-experiment-tracking.db