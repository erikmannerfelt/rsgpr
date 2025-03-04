{ pkgs ? import <nixpkgs> { }, gdal ? null }:
let
  # The rust gdal system is a bit hard to compile without precompiled bindings.
  # Sometimes the unstable version of GDAL is updated faster than the precompiled bindings are in gdal-sys.
  # Therefore, it may be necessary to provide an older GDAL as an argument. This "hack" is kept here in case
  # it becomes needed in the future.
  my-gdal = if gdal != null then gdal else pkgs.gdal;
  manifest = (pkgs.lib.importTOML ./Cargo.toml).package;

in
pkgs.rustPlatform.buildRustPackage rec {

  pname = manifest.name;
  version = manifest.version;

  cargoLock.lockFile = ./Cargo.lock;
  nativeBuildInputs = with pkgs; [
    cargo # Manage rust projects`
    rustc # Compile rust
    pkg-config # Needed to install gdal-sys
    cmake
    gnumake
    clang
    zlib
  ];
  buildInputs = with pkgs; [ proj my-gdal ];

  # libz-sys fails when compiling, so testing is disabled for now
  doCheck = false;

  src = pkgs.lib.cleanSource ./.;

}
