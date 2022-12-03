{
  description = "rsgpr development shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # 22.05 is used for GDAL 3.5 (see more in 'shell.nix')
    nixpkgs-2205.url = "github:NixOS/nixpkgs/nixos-22.05";
    flake-utils.url = "github:numtide/flake-utils";
  };
  
  outputs = {self, nixpkgs, flake-utils, nixpkgs-2205}: 
    flake-utils.lib.eachDefaultSystem 
      (system: 
        let 
          pkgs = nixpkgs.legacyPackages.${system}; 
          old-pkgs = nixpkgs-2205.legacyPackages.${system};
        in 
        {
          devShells.default = import ./shell.nix {inherit pkgs; gdal=old-pkgs.gdal;};
        }
      
      );
}