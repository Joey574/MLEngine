{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  nativeBuildInputs = with pkgs; [
    cmake
    ninja
    pkg-config
  ];

  buildInputs = with pkgs; [
    openblas
    bc
  ];

  shellHook = ''
    export NIX_ENFORCE_NO_NATIVE=0
  '';
}
