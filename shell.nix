{ pkgs ? import <nixpkgs> {}, gdal ? null }:

let 
  # The rust gdal system is a bit hard to compile without precompiled bindings.
  # Sometimes the unstable version of GDAL is updated faster than the precompiled bindings are in gdal-sys.
  # Therefore, it may be necessary to provide an older GDAL as an argument. This "hack" is kept here in case
  # it becomes needed in the future.
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
