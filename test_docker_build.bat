@echo off
REM Script pour tester le build Docker localement avant Railway

echo ========================================
echo TEST BUILD DOCKER LOCAL
echo ========================================
echo.

REM Se placer à la racine du projet
cd /d "%~dp0.."

echo [1/4] Nettoyage des anciens containers...
docker ps -a -q --filter "name=fraud-test" | findstr . && docker rm -f fraud-test

echo.
echo [2/4] Build de l'image (contexte = racine du projet)...
docker build -t fraud-detection-test -f APP_autoML\Dockerfile .

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ ERREUR: Le build a échoué!
    pause
    exit /b 1
)

echo.
echo ✅ Build réussi!
echo.
echo [3/4] Test de la structure de l'image...
docker run --rm fraud-detection-test ls -la /
docker run --rm fraud-detection-test ls -la /automl_transformer
docker run --rm fraud-detection-test ls -la /data/metatransformer_training

echo.
echo [4/4] Voulez-vous lancer le container en mode test? (Y/N)
set /p LAUNCH=
if /i "%LAUNCH%"=="Y" (
    echo Lancement du container sur http://localhost:5001...
    echo Variables d'environnement chargées depuis .env
    docker run --name fraud-test -p 5001:5000 --env-file APP_autoML\.env fraud-detection-test
)

echo.
echo ========================================
echo Nettoyage: docker rm fraud-test
echo Image: docker rmi fraud-detection-test
echo ========================================
pause
