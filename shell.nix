{ pkgs ? import <nixpkgs> {} }:

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
       gdal
  ];
  LIBCLANG_PATH = "${pkgs.libclang.lib}/lib";
  PKG_CONFIG_PATH="${pkgs.gdal}/lib/pkgconfig/:${pkgs.proj.dev}/lib/pkgconfig";
  BINDGEN_EXTRA_CLANG_ARGS="-I ${pkgs.proj.dev}/include -I ${pkgs.clang}/resource-root/include";

  shellHook = ''
      ${pkgs.zsh}/bin/zsh
      alias rsgpr="$(pwd)/target/debug/rsgpr";
  '';
}
