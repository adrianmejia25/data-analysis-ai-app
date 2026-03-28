import pandas as pd

def cargar_archivo(ruta):
    # se carga el archivo dependiendo de la extension que tenga
    if ruta.endswith(".csv"):
      # pd es como se le llama a la librería Pandas, el read_csv es "leer archivo separado por comas"
        data_frame = pd.read_csv(ruta)
    elif ruta.endswith(".xlsx") or ruta.endswith(".xls"):
        data_frame = pd.read_excel(ruta)
    else:
        raise ValueError("Solamente se permite formato: .csv, .xlsx o .xls")

    # con este drop eliminamos las filas que estén exactamente repetidas
    data_frame = data_frame.drop_duplicates()

    # No se dejan valores nulos, se llena con 0 cuando el tipo de dato de esa columna es número, y si es texto lo llenamos con "Desconocido"
    for columna in data_frame.columns:
        if data_frame[columna].dtype in ["int64", "float64"]:
            data_frame[columna] = data_frame[columna].fillna(0)
        else:
            data_frame[columna] = data_frame[columna].fillna("Desconocido")

    # se intenta detectar las fechas automaticamente y se busca nada mas en las columnas que son de tipo objeto(texto)  
    for columna in data_frame.select_dtypes(include="object").columns:
        try:
          # pd.to_datetime trata de convertir el texto a formato de fecha real
            data_frame[columna] = pd.to_datetime(data_frame[columna], format="mixed", dayfirst=False)
        except Exception:
          # en caso de que no sea una fecha por ejemplo el nombre del restaurante, simplemente no hace nada y sigue
            pass

    return data_frame