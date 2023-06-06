{
  description = "rsgpr development shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.default = import ./shell.nix { inherit pkgs; };
        packages = rec {
          rsgpr = import ./default.nix { inherit pkgs; };
          default = rsgpr;
        };
      }

    );
}
