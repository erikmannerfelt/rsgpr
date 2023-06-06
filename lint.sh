#!/usr/bin/env bash
set -e

cargo fmt --check
cargo check -- -Dwarnings
cargo clippy -- -Dwarnings

echo "hello"
