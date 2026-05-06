import subprocess
import sys
from pathlib import Path

def ejecutar_script(python_exe, ruta_script):
    print(f"\n>>> Ejecutando: {ruta_script.name}")
    try:
        subprocess.run(
            [str(python_exe), str(ruta_script)],
            check=True,
            cwd=str(ruta_script.parent),
        )
        print(f">>> Finalizado: {ruta_script.name}")
    except subprocess.CalledProcessError as e:
        print(f"!!! Error en {ruta_script.name}: {e}")
        sys.exit(1)

def main():
    # 1. DETECCIÓN AUTOMÁTICA DEL EJECUTABLE
    # Si ya estás en el ambiente activo, sys.executable es la ruta correcta.
    PYTHON_EXE = Path(sys.executable)
    
    print(f"--- Usando Python de: {PYTHON_EXE} ---")

    BASE_DIR = Path(__file__).resolve().parent
    
    # 2. LISTA DE SCRIPTS
    scripts = [
        BASE_DIR / "MINERful" / "modelo_declarativo_enriquecedio_v2.py",
        BASE_DIR / "dg_training.py",
        BASE_DIR / "dg_prediction.py"
    ]
    
    for script in scripts:
        if script.exists():
            ejecutar_script(PYTHON_EXE, script)
        else:
            print(f"Error: No se encontró el archivo {script}")
            sys.exit(1)

if __name__ == "__main__":
    main()