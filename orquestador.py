import subprocess
import sys
from pathlib import Path

def ejecutar_script(python_exe, ruta_script):
    """Ejecuta un script usando un ejecutable de Python específico."""
    print(f"\n>>> Iniciando en 'deep_generator': {ruta_script.name}")
    try:
        # Usamos el path del ambiente definido, no sys.executable
        subprocess.run([str(python_exe), str(ruta_script)], check=True)
        print(f">>> Finalizado con éxito: {ruta_script.name}")
    except subprocess.CalledProcessError as e:
        print(f"!!! Error en {ruta_script.name}: {e}")
        sys.exit(1)

def main():
    BASE_DIR = Path(__file__).resolve().parent
    
    # --- CONFIGURACIÓN DEL AMBIENTE ---
    # Cambia esta ruta por la ubicación real de tu ambiente deep_generator
    # Ejemplo típico en Conda: r"C:\Users\Diego\anaconda3\envs\deep_generator\python.exe"
    PYTHON_DEEP_GEN = Path(r"C:\Ruta\A\Tu\Ambiente\deep_generator\python.exe")
    
    if not PYTHON_DEEP_GEN.exists():
        print(f"Error: No se encontró el ejecutable en {PYTHON_DEEP_GEN}")
        return

    scripts = [
        BASE_DIR / "MINERful" / "modelo_declarativo_enriquecedio_v2.py",
        BASE_DIR / "dg_training.py",
        BASE_DIR / "dg_prediction.py"
    ]
    
    for script in scripts:
        if script.exists():
            ejecutar_script(PYTHON_DEEP_GEN, script)
        else:
            print(f"Error: Archivo no encontrado en {script}")
            break

if __name__ == "__main__":
    main()