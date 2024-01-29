#!/bin/bash

# NOTE: 
# This script is used to setup a substrate node on a fresh ubuntu/debian 20.04 machine.
# Any other OS or version may not work as expected.

# Pull gh 
git clone https://github.com/opentensor/subtensor.git
cd subtensor

# Setup Rust
sudo apt update
sudo apt install -y git clang curl libssl-dev llvm libudev-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

rustup default stable
rustup update
rustup update nightly
rustup target add wasm32-unknown-unknown --toolchain nightly

# Use Rust's native cargo command to build and launch the template node. 
cargo run --release -- --dev
cargo build --release

# Single-node development chain with non-persistent state.
./target/release/subtensor --dev
./target/release/subtensor purge-chain --dev

#Start the development chain with detailed logging.
RUST_BACKTRACE=1 ./target/release/subtensor-ldebug --dev
SKIP_WASM_BUILD=1 RUST_LOG=runtime=debug -- --nocapture

