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
    proj
    gdal
  ];

  shellHook = ''
    ${pkgs.zsh}/bin/zsh
    alias rsgpr="$(pwd)/target/debug/rsgpr";
  '';
}
