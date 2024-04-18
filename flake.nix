{
  description = "Simple (-ish) Ground Penetrating Radar software";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          rsgpr = import ./default.nix { inherit pkgs; };

        in
        {
          devShells.default = import ./shell.nix { inherit pkgs; };
          defaultPackage = rsgpr;
          packages = {
            inherit rsgpr;
            default = rsgpr;
          };
        }

      ) // {
      overlays.default = final: prev: {
        rsgpr = import ./default.nix { pkgs = final; };
      };
    };
}
