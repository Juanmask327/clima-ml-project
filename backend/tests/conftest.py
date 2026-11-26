"""
Configuración de pytest: permite importar 'api.main'
ejecutando los tests desde cualquier directorio.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
