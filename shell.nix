{ pkgs ? import <nixpkgs> { }, gdal ? null }:

let
  package = import ./default.nix { inherit pkgs gdal; };

in
pkgs.mkShell {

  inputsFrom = [ package ];

  buildInputs = with pkgs; [
    cargo-tarpaulin # Get test coverage statistics
    rustfmt
    clippy
  ];
  LIBCLANG_PATH = package.LIBCLANG_PATH;
  PKG_CONFIG_PATH = package.PKG_CONFIG_PATH;
  BINDGEN_EXTRA_CLANG_ARGS = package.BINDGEN_EXTRA_CLANG_ARGS;

  shellHook = ''
    ${pkgs.zsh}/bin/zsh
    alias rsgpr="$(pwd)/target/debug/rsgpr";
  '';
}
