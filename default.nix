{ pkgs ? import <nixpkgs> { } }:
{
  glados = pkgs.python3Packages.buildPythonApplication {
    name = "glados";
    src = ./.;
    propagatedBuildInputs = with pkgs.python3Packages; [
      inflect
      (
        # XXX phonemizer is only available as application
        phonemizer.override {
          buildPythonApplication = buildPythonPackage;
        }
      )
      pytorch
      scipy
      unidecode
    ];
    postInstall = ''
      mv $out/bin/glados.py $out/bin/glados
      mkdir -p $out/share
      cp -r $src/models $out/share
    '';
  };
}
