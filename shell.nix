{ pkgs ? import <nixpkgs> {}, gdal ? null }:

let 
  # The rust gdal system is a bit hard to compile without precompiled headers.
  # So the minimum supported gdal version is currently 3.5 (2022-11-03)
  my-gdal = if gdal != null then gdal else pkgs.gdal;

in

pkgs.mkShell {

  buildInputs = with pkgs; [
       cargo  # Manage rust projects`
       cargo-tarpaulin  # Get test coverage statistics
       rustc  # Compile rust
       pkg-config # Needed to install gdal-sys
       cmake
       gnumake
       proj
       proj.dev
       libclang
       clang
       my-gdal
       rustfmt

  ];
  LIBCLANG_PATH = "${pkgs.libclang.lib}/lib";
  PKG_CONFIG_PATH="${pkgs.gdal}/lib/pkgconfig/:${pkgs.proj.dev}/lib/pkgconfig";
  BINDGEN_EXTRA_CLANG_ARGS="-I ${pkgs.proj.dev}/include -I ${pkgs.clang}/resource-root/include -I ${pkgs.gcc}/include";

  shellHook = ''
      ${pkgs.zsh}/bin/zsh
      alias rsgpr="$(pwd)/target/debug/rsgpr";
  '';
}
