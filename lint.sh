#!/usr/bin/env bash
set -e

echo "Running pre-commit"
pre-commit run --all-files
echo "Running cargo fmt"
cargo fmt --check
echo "Running cargo clippy"
cargo clippy -- -Dwarnings
