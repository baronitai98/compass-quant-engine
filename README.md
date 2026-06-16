# Compass quant-engine

Microservicio de optimización de cartera (FastAPI + Ledoit-Wolf + PyPortfolioOpt) que tu app de Lovable llama vía `optimizer.functions.ts`. Implementa el contrato exacto que ese cliente ya espera, así que **no tienes que tocar el código de Lovable**: solo desplegar esto y poner dos variables de entorno.

## Qué hace

`POST /optimize` (header `x-secret`): recibe la matriz de retornos T×N, los Conviction Scores como "vistas", y los límites de posición/sector; estima la covarianza con **Ledoit-Wolf**, resuelve **media-varianza** (máxima utilidad cuadrática) con restricciones lineales y una penalización L2 para diversificar, y devuelve los pesos. Si el problema es infactible, degrada a mínima volatilidad y, en último caso, a un reparto determinista que respeta los caps (nunca devuelve algo que viole un cap).

`GET /health`: para el health check del hosting y para el "ping de despertar" del cron de la Fase 8.

## Desplegar (Render — recomendado, free tier)

1. Sube esta carpeta a un repo de GitHub (público o privado).
2. En **Render → New → Web Service**, conecta el repo. Render detecta `render.yaml`. Si no:
   - Runtime: **Python 3**
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Health check path: `/health`
3. En **Environment**, agrega:
   - `QUANT_SHARED_SECRET` = un secreto largo aleatorio (guárdalo).
4. Deploy. Copia la URL pública (p.ej. `https://compass-quant-engine.onrender.com`).

### Alternativa: Railway
New Project → Deploy from repo → agrega la variable `QUANT_SHARED_SECRET` → Railway usa el `Procfile` automáticamente.

## Conectar con Lovable

En **Lovable Cloud → variables de entorno del servidor** agrega (NO en el cliente):

- `QUANT_ENGINE_URL` = la URL pública del servicio (sin barra final).
- `QUANT_SHARED_SECRET` = el mismo secreto que pusiste en el hosting.

## Verificar que quedó vivo

```bash
curl https://TU-URL/health
# -> {"ok":true,"service":"quant-engine","version":"1.0.0"}
```

Luego, en la app: genera una propuesta en `/proposal`. El campo `modelSource` debe pasar de `"rules"` a `"optimizer"`. Si sigue en `"rules"`, revisa: (a) que ambas env vars estén en Lovable, (b) que el servicio no esté "dormido" (free tier duerme tras inactividad; el primer request tarda y puede superar el timeout de 8s del cliente — vuelve a intentar, o deja que el cron diario lo despierte).

## Nota de costo / arranque en frío

El free tier de Render/Railway suspende el servicio tras ~15 min de inactividad; el primer request luego tarda decenas de segundos (arranque en frío de cvxpy). Opciones: subir a un plan que no duerma, o usar el ping diario del cron de la Fase 8 para mantenerlo caliente en horario relevante.

## Extensión futura (Black-Litterman)

Hoy las vistas de conviction se mapean a un vector de retornos esperados centrado/escalado (±8% anual). Para Black-Litterman completo (Π de equilibrio, Ω/τ, confianzas de Idzorek), `pypfopt` trae `BlackLittermanModel`; se puede añadir un parámetro `method: "mvo" | "bl"` al body sin romper el contrato actual.
