#!/bin/sh

rm -rf mlflow-experiment-tracking.db

sqlite3 mlflow-experiment-tracking.db <<'END_SQL'
.read ./database-state.sql
END_SQL