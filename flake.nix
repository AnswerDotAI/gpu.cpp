{
  description = "gpu.cpp";

  nixConfig = {
    bash-prompt = "\[gpu.cpp$(__git_ps1 \" (%s)\")\]$ ";
  };

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import nixpkgs {
            inherit system;
          };
        in
        with pkgs;
        {
          devShells.default = mkShell ({
            shellHook = ''
              source ${git}/share/bash-completion/completions/git-prompt.sh
              if [ -f /run/opengl-driver/share/vulkan/icd.d/nvidia_icd.x86_64.json ] ; then
                export VK_DRIVER_FILES=/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.x86_64.json
              fi
            '';
            buildInputs = [
              cmake
              clang
              python3
              git
            ] ++ lib.optionals stdenv.isLinux [
              xorg.libX11.dev
              xorg.libXrandr.dev
              xorg.libXinerama.dev
              xorg.libXcursor.dev
              xorg.libXi.dev
              gdb
            ];
          } // (if stdenv.isLinux then {
            LD_LIBRARY_PATH="${vulkan-loader}/lib";
          } else {}));
        }
      );
}
