#!/usr/bin/env bash
set -e

pre-commit run --all-files
cargo fmt --check
cargo clippy -- -Dwarnings
