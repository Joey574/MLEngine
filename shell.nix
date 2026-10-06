{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  buildInputs = with pkgs; [
    cmake
    ninja
    yaml-cpp
  ];

  shellHook = ''
    export NIX_ENFORCE_NO_NATIVE=0
  '';
}
