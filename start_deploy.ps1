# ============================================================
# PREDICCIÓN CLIMÁTICA ML · Arranque del despliegue local
# ============================================================
# Levanta la app (contenedor Docker) y la publica en Internet
# mediante un túnel de Cloudflare.
#
#   .\start_deploy.ps1
#
# Requisitos:
#   - Docker Desktop arrancado
#   - cloudflared instalado (winget install Cloudflare.cloudflared)
# ============================================================

$ErrorActionPreference = 'Stop'
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Container  = 'clima-ml-test'
$Port       = 8099

Write-Host "`n=== PREDICCION CLIMATICA ML · Despliegue local ===`n" -ForegroundColor Cyan

# ── 1. Construir la imagen ─────────────────────────────────
Write-Host "[1/4] Construyendo imagen Docker..." -ForegroundColor Yellow
Push-Location $ProjectDir
docker build -t clima-ml:test . | Out-Host
Pop-Location

# ── 2. Levantar el contenedor ──────────────────────────────
Write-Host "`n[2/4] Levantando contenedor..." -ForegroundColor Yellow
docker rm -f $Container 2>$null | Out-Null
docker run -d --name $Container -p "${Port}:8000" clima-ml:test | Out-Host

Write-Host "    Esperando a que el modelo cargue..."
for ($i = 0; $i -lt 30; $i++) {
  Start-Sleep -Seconds 2
  try {
    $h = Invoke-RestMethod "http://localhost:${Port}/health" -TimeoutSec 5
    if ($h.model_loaded) { Write-Host "    OK - modelo cargado" -ForegroundColor Green; break }
  } catch {}
}

# ── 3. Verificar la API ────────────────────────────────────
Write-Host "`n[3/4] Verificando endpoints..." -ForegroundColor Yellow
foreach ($p in @('/health', '/', '/model/info')) {
  $code = curl.exe -s -m 15 -o NUL -w "%{http_code}" "http://localhost:${Port}$p"
  Write-Host "    $p -> HTTP $code"
}

# ── 4. Túnel público ───────────────────────────────────────
Write-Host "`n[4/4] Abriendo túnel público (Cloudflare)..." -ForegroundColor Yellow
$cf = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
if (-not $cf) { $cf = "C:\Program Files (x86)\cloudflared\cloudflared.exe" }

if (Test-Path $cf) {
  $log = "$env:TEMP\cf_tunnel.txt"
  Remove-Item $log -ErrorAction SilentlyContinue
  Start-Process -FilePath $cf `
    -ArgumentList 'tunnel', '--url', "http://localhost:${Port}", '--no-autoupdate' `
    -RedirectStandardOutput $log -RedirectStandardError "$env:TEMP\cf_tunnel_err.txt" -NoNewWindow

  Start-Sleep -Seconds 15
  $url = (Select-String -Path "$env:TEMP\cf_tunnel_err.txt" -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' -ErrorAction SilentlyContinue |
          Select-Object -First 1).Matches.Value

  if ($url) {
    Write-Host "`n  =============================================" -ForegroundColor Green
    Write-Host "   APP PUBLICA: $url" -ForegroundColor Green
    Write-Host "  =============================================`n" -ForegroundColor Green
    Write-Host "  El tunel corre mientras la maquina este encendida."
    Write-Host "  Para la URL publica permanente, usa Render (ver readme.md)."
  } else {
    Write-Host "    No se pudo obtener la URL (revisa $env:TEMP\cf_tunnel_err.txt)" -ForegroundColor Red
  }
} else {
  Write-Host "    cloudflared no encontrado. El tunel no se puede abrir." -ForegroundColor Red
}

Write-Host "`nPresiona Enter para cerrar (el contenedor y el tunel siguen corriendo).`n" -ForegroundColor DarkGray
$null = Read-Host
