# Entregable 2 — Segmentación semántica con U-Net

Proyecto de **Aprendizaje Profundo** (Grado en Ciencia de Datos, UPNA) — Grupo "<X>".

Pipeline de segmentación semántica sobre imágenes de productos de supermercado (dataset D2S).
Dada una imagen RGB, el modelo predice para cada píxel la clase a la que pertenece
(`0` = fondo, `1..C` = producto).

- **Objetivo 1 (6 pts):** versión inicial con una U-Net compacta y entrenamiento estándar.
- **Objetivo 2 (hasta 4 pts):** propuestas de mejora (datos, arquitectura, función de coste,
  optimización…) justificadas y comparadas contra la versión inicial.

## Equipo



## Estructura del proyecto

```
entregable2-unet/
├── .vscode/launch.json        <- Configuraciones para lanzar scripts desde VSCode
├── config/                    <- Ficheros de configuración de experimentos
├── data/                      <- Datasets (NO se suben a GitHub, ver abajo)
├── reports/
│   ├── models/                <- Modelos entrenados (.pth) — no se suben
│   ├── results/               <- Métricas y resultados — no se suben
│   └── runs/                  <- Logs de TensorBoard — no se suben
├── runs/
├── src/
│   ├── data/
│   │   └── dataset_semseg.py  <- Clase SupermarketSemSeg (carga imagen + máscara)
│   ├── models/
│   │   └── layers/            <- Capas propias
│   ├── runs/
│   │   └── main.py            <- Punto de entrada del proyecto
│   ├── utils/                 <- Funciones auxiliares (métricas, rutas, etc.)
│   └── visualization/         <- Gráficas y visualización de predicciones
├── requirements.txt
└── README.md
```

## Puesta en marcha

### 1. Clonar el repositorio

En VSCode: `View → Command Palette → Git: Clone → Clone from GitHub` y seleccionar este repositorio.

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Colocar los datos

Los datos **no están en el repositorio** (pesan demasiado). Cada miembro debe descargarlos
de miAulario y descomprimirlos dentro de `data/`, de forma que quede así:

```
data/
├── dataset_res_144_192_10classes/
│   ├── label_mapping_10classes.json
│   ├── train/{images,masks}/
│   └── test/{images,masks}/
├── dataset_res_144_192/              <- versión con 60 clases
│   ├── label_mapping.json
│   ├── train/{images,masks}/
│   └── test/{images,masks}/
└── (opcional) versiones 360x480
```

| Versión | Train | Test | Clases (sin fondo) |
|---------|-------|------|--------------------|
| `dataset_res_144_192_10classes` | 630 | 1290 | 10 |
| `dataset_res_144_192` | 4380 | 3600 | 60 |


> ⚠️ Nunca hagáis commit de ficheros dentro de `data/`. El `.gitignore` ya los excluye;
> si alguna vez aparecen imágenes o `.zip` en *Source Control*, no hagáis commit y avisad.

## Ejecución

**Desde VSCode:** panel *Run and Debug* → configuración **"Lanzar proyecto"**
(ejecuta `src/runs/main.py` con los argumentos definidos en `.vscode/launch.json`).

**Desde terminal:**

```bash
cd src/runs
python main.py
```

> El directorio de trabajo es `src/runs`, por lo que las rutas a `data/` y `reports/`
> deben construirse a partir de la raíz del proyecto
> (`Path(__file__).resolve().parents[2]`), no con rutas relativas como `'./data'`.

### TensorBoard

```bash
tensorboard --logdir reports/runs
```

Cada ejecución guarda sus logs en una subcarpeta `reports/runs/<fecha_hora>/`.

### Flujo de trabajo con Git

Todos los comandos se ejecutan en la terminal de VSCode (`View → Terminal`) desde la raíz
del proyecto (`entregable2-unet/`). Entre paréntesis, el equivalente con botones de VSCode.

### Configuración inicial (una sola vez por ordenador)

```bash
git config --global user.name "Nombre Apellido"
git config --global user.email "correo-de-tu-cuenta-github"
git config --global pull.rebase false
git config --global --list          # comprobar que se ha guardado
```

### Rutina de cada sesión

**1. Al empezar: traer los cambios de los demás**

```bash
git pull                            # (Source Control → Sync Changes)
```

**2. Trabajar.** Preferentemente en vuestros propios archivos, para evitar conflictos.

**3. Revisar qué has cambiado**

```bash
git status                          # lista de archivos modificados (M), nuevos (U/??), borrados (D)
git diff                            # detalle línea a línea de los cambios
git diff src/models/unet.py         # detalle de un archivo concreto
```

**4. Preparar los archivos para el commit (staging)**

```bash
git add src/models/unet.py          # añadir un archivo concreto   (botón + junto al archivo)
git add .                           # añadir todos los cambios     (botón + junto a "Changes")
git status                          # comprobar que en "Changes to be committed" está lo que quieres
```

> ⚠️ Antes de `git add .`, revisa `git status`: no debe aparecer nada de `data/`, ni
> imágenes, ni `.zip`, ni `.pth`.

**5. Hacer commit (se guarda solo en tu ordenador)**

```bash
git commit -m "Añadida métrica mIoU"   # (escribir mensaje → Commit)
```

**6. Al terminar: subir tus cambios**

```bash
git pull                            # primero traer lo último, por si alguien ha subido algo
git push                            # (Sync Changes)
```

### Comprobar si hay cambios nuevos sin tocar tus archivos

```bash
git fetch                           # mira si hay cambios en GitHub
git status                          # "Your branch is behind 'origin/main' by 2 commits" → hay que hacer pull
```

En la barra de estado de VSCode: `↓2 ↑0` = hay 2 commits en GitHub que no tienes;
`↓0 ↑1` = tienes 1 commit sin subir.

### Ver el historial

```bash
git log --oneline                   # lista de commits, uno por línea
git log --oneline -- src/data/      # solo los commits que tocaron esa carpeta
```

### Deshacer cambios

```bash
git restore src/models/unet.py      # descartar cambios NO commiteados de un archivo
                                    # (flecha curva "Discard Changes")
git restore --staged src/models/unet.py   # sacar un archivo del staging sin perder los cambios
```

> ⚠️ `git restore <archivo>` borra tus cambios locales desde el último commit y **no se
> puede deshacer**.

### Resolver un conflicto

Aparece al hacer `git pull` si tú y otra persona habéis cambiado las mismas líneas:

```
CONFLICT (content): Merge conflict in src/runs/main.py
```

1. Ver qué archivos están en conflicto:
```bash
   git status                       # aparecen en "Unmerged paths" (en VSCode, con una C)
```
2. Abrir el archivo y elegir qué versión mantener:
```
   <<<<<<< HEAD
   tu versión               (Current Change)
   =======
   versión de GitHub        (Incoming Change)
   >>>>>>> 3f2a9c1
```
   Usar *Accept Current / Accept Incoming / Accept Both*, o editar a mano borrando las
   líneas `<<<<<<<`, `=======` y `>>>>>>>`. Guardar el archivo.
3. Marcarlo como resuelto, cerrar el merge y subir:
```bash
   git add src/runs/main.py
   git commit                       # el mensaje de merge ya viene escrito (salir del editor con :wq si se abre vim)
   git push
```

Si te lías a mitad del conflicto y quieres volver al estado anterior al `pull`:

```bash
git merge --abort
```

### Error habitual

```
error: Your local changes to the following files would be overwritten by merge
```

Significa que has intentado hacer `pull` con cambios sin commitear. Solución: hacer
commit primero (pasos 4 y 5) y repetir `git pull`.

### Normas del grupo

- **Pull al empezar, pull + push al terminar.**
- Commits pequeños con mensajes que digan qué se ha hecho.
- El código compartido va en `.py` dentro de `src/`. Los notebooks son personales
  (`notebooks/<nombre>_<tema>.ipynb`), porque generan conflictos difíciles de resolver.
- No subir datos, modelos (`.pth`) ni logs de TensorBoard.

Guía detallada: `Chuleta_Git.pdf` (miAulario).