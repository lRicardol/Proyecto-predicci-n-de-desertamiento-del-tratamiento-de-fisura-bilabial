# Predicción de deserción del tratamiento de fisura bilabial

Sistema de analítica predictiva y conversacional sobre deserción terapéutica en pacientes con fisura labiopalatina. La arquitectura sigue **Clean Architecture + SOLID**: el dominio solo depende de interfaces; CSV, XGBoost, LangChain y FastAPI viven en infraestructura y aplicación.

## Cómo ejecutar

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python main.py
```

- Panel: `http://127.0.0.1:8000/`
- Predicción GET: `http://127.0.0.1:8000/api/predictions/PAC-1001`
- Predicción POST: `http://127.0.0.1:8000/api/predictions`
- Analítica: `POST /api/analytics` con `{"query": "Grafica la relación entre inasistencias y deserción"}`
- OpenAPI: `http://127.0.0.1:8000/docs`

Sin `OPENAI_API_KEY` el analista genera gráficos locales con matplotlib sobre `data/raw/dataset_desercion_fisura_bilabial.csv`. Con clave, usa un agente pandas de LangChain.

## Datos

Los CSV reales de `data/raw/` se consolidan por `ID_Paciente` (cohortes PAC-1000 y PAC-2000): deserción socioeconómica, hitos quirúrgicos y terapia de lenguaje.

## Pruebas

```bash
pytest -q
```
