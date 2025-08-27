## RODAR MLFLOW LOCAL 
# Estou utilizando python 3.10.13

# %%
# Instalar bibliotecas necessárias:
# %pip install mlflow

# Importar bibliotecas
import webbrowser
import subprocess
import time
import mlflow

# Define o diretório raiz
mlflow.set_tracking_uri(
    "file:///D:/OneDrive/Documentos/GitHub/portifolio/aplicacao_mlflow/mlflow-server"
)


# Função para facilitar abertura e fechamento do mlflow
def start_mlflow(port=5000):
    """Inicia MLflow UI"""
    process = subprocess.Popen(
        ["mlflow", "ui", "--port", str(port)], 
        shell=True
    )
    time.sleep(3)
    webbrowser.open(f"http://127.0.0.1:{port}")
    print(f"MLflow iniciado na porta {port} (PID: {process.pid})")
    return process

def stop_mlflow_by_port(port=5000):
    """Para MLflow pela porta (mais confiável)"""
    try:
        # Encontra e mata o processo pela porta
        result = subprocess.run(
            f'netstat -ano | findstr ":{port}"',
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if 'LISTENING' in line:
                    pid = line.split()[-1]
                    subprocess.run(f"taskkill /F /T /PID {pid}", shell=True)
                    print(f"MLflow na porta {port} finalizado (PID: {pid})")
                    return True
        
        print(f"Nenhum processo MLflow encontrado na porta {port}")
        return False
        
    except Exception as e:
        print(f"Erro: {e}")
        return False

# Uso simples:
process = start_mlflow(5000)

# %%

# Para fechar:
stop_mlflow_by_port(5000)


