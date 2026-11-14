# 🧺 Ironbox Garment & Ironing Computer Vision Dataset

> **Project Owner:** Dakshan  
> **Start Date:** November 2026  
> **Last Updated:** 2026-11-14 19:15 IST  
> **Previous Updates:** 2026-11-13 11:15 IST | 2026-11-13 10:56 IST | 2026-11-12 20:36 IST | 2026-11-12 20:23 IST | 2026-11-12 20:22 IST | 2026-11-12 20:08 IST | 2026-11-12 18:56 IST | 2026-11-12 18:37 IST | 2026-11-12 10:26 IST | 2026-11-12 10:22 IST | 2026-11-12 10:21 IST | 2026-11-12 10:11 IST  
> **Phase 1 Status:** Finished on 2026-11-12 — Image Classification, Folder Segregation, Dataset Indexing, and Remote GitHub Push (337 files, 1.17 GB, 34 garment categories)  
> **Current Status:** Phase 2 — Dataset Annotation, Computer Vision Preprocessing, Feature Extraction & Training Pipeline Formulation  

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Design Evolution & Decisions](#design-evolution--decisions)
- [Dataset Bill of Materials / Inventory Specs](#dataset-bill-of-materials--inventory-specs)
- [Dataset Architecture & Folder Taxonomy](#dataset-architecture--folder-taxonomy)
- [Capture Environment & Physical Setup](#capture-environment--physical-setup)
- [Dataset Taxonomy & Multidimensional Variety](#dataset-taxonomy--multidimensional-variety)
- [Software & Tooling Stack](#software--tooling-stack)
- [Current Progress](#current-progress)
- [Main To-Do List](#main-to-do-list)
- [Known Issues & Solutions](#known-issues--solutions)
- [Build Phases](#build-phases)
- [Information Log (Chronological)](#information-log-chronological)
- [📐 Rules for AI Assistant](#-rules-for-ai-assistant)
- [Reference Links](#reference-links)
- [How to Continue in a New Chat Session](#how-to-continue-in-a-new-chat-session)

---

## Project Overview

A **specialized high-resolution computer vision dataset** capturing real-world textile manipulation, fabric wrinkling, water-sprinkling pre-treatment, and thermodynamic heat-pressing actions on an authentic Indian street-side mobile ironing cart (*dhobi cart*) utilizing a heavy charcoal brass iron box.

The dataset serves as ground-truth image data for tasks such as:
1. **Garment Re-Identification & Instance Tracking**: Tracking individual pieces of clothing through intense physical transformations (wrinkled $\to$ wet-droplet stage $\to$ heat-flattened $\to$ origami-style retail fold).
2. **Fabric & Material Classification**: Discerning subtle optical, tactile, and reflectance properties across 10 diverse textile categories (from diaphanous crinkle tissue and slub linen to heavy cotton twill chinos and poly-viscose suiting).
3. **State & Workflow Phase Estimation**: Detecting procedural stages (dry wrinkled, water-sprinkled, hot-pressed body, collar/crease formation, folded finished pile).

### Key Specs (Dataset Snapshot)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Total Files | 317 files | 314 active usable images + 3 markdown docs |
| Clean Usable Images | **314 photos** | 304 garment close-ups + 10 context/batch reference photos (100% of media) |
| Image Resolution | **3072 × 4096 px** (12.6 MP) | Standard 4:3 high-definition camera capture |
| Total Size | **~1.13 GB** | High-resolution original quality |
| Total Garments | **34 unique garments** | Spanning women's ethnic, men's formal, casual, & bottoms |
| Distinct Fabric Types | **10 material classes** | Pure cotton, linen, rayon, PV suiting, knit jersey, etc. |
| Visual Patterns | **8 surface patterns** | Solid, floral, plaid, windowpane check, zigzag, stripes, etc. |
| Git Repository | [Dakshankarthic/iron](https://github.com/Dakshankarthic/iron) | Branch: `main` |

---

## Design Evolution & Decisions

### 1. Workspace Organization *(decided 2026-11-12, updated 2026-11-14)*

| Original State | Final Organized Architecture |
|----------------|------------------------------|
| Flat root dump of 335 files in `I:\ironbox\` | **34 numbered garment folders (`01_...` to `34_...`)** |
| Mix of active photos, bursts, and Android `.trashed` | **Noise & trashed files purged; clean supporting context preserved in `_cart_setup` and `_batch_stacks`** |
| No metadata or index | **Handcrafted `README.md`, `DATASET_INDEX.md`, and `rule_book_dk.md` at root** |

**Rationale:** Clean modular folders allow downstream computer vision training loops (`torchvision.datasets.ImageFolder` or Hugging Face `datasets`) to ingest classes directly without name collision or path ambiguity.

### 2. Multi-Stage Garment Association *(decided 2026-11-12)*

- The ironing process features an initial **water-sprinkling batch** (frames `099` to `109`) where the worker sprays water on shirts before ironing.
- The worker also performed **interleaved pressing** (e.g. pressing part of a peach shirt, switching to a kurti neckline, and returning to the peach shirt).
- **Decision:** Rather than splitting the temporal sequence into fragmented sub-folders, **all images belonging to the same physical garment** (including the pre-iron sprinkling shot and both pressing phases) are grouped into that garment's single folder.

### 3. Remote Synchronization & Clean State *(updated 2026-11-14)*

- **Target Repository:** `https://github.com/Dakshankarthic/iron.git`
- Initialized on branch `main` with 314 clean usable photos and zero noise/trash files.

---

## Dataset Bill of Materials / Inventory Specs

### Raw Capture Media

| Component | Quantity | Specs | Notes |
|-----------|:--------:|-------|-------|
| Active Garment Photos | 304 | JPEG format, 3072 × 4096 px | High-resolution close-ups of 34 garments being ironed |
| Context Photos | 6 | JPEG format, 3072 × 4096 px | Street cart environment, charcoal box, worker |
| Stack Photos | 4 | JPEG format, 3072 × 4096 px | Finished folded clothing stack reference shots |
| **Total Usable Images** | **314** | **100% clean image dataset** | **Zero noise or deleted files** |

---

## Dataset Architecture & Folder Taxonomy

### Directory Tree

```
I:\ironbox\
├── 01_white_embroidered_salwar/     ← 2 photos (White cambric bottoms with hem eyelets)
├── 02_red_salwar/                   ← 6 photos (Crimson red cotton salwar bottoms)
├── 03_beige_peach_salwar/           ← 11 photos (Beige/peach soft cotton salwar)
├── 04_dark_printed_kurti/           ← 5 photos (Charcoal kurti with booti motifs)
├── 05_white_pink_floral_kurti/      ← 25 photos (Slub linen kurti with resham embroidery)
├── 06_lavender_floral_kurti/        ← 17 photos (Lilac botanical print rayon kurti)
├── 07_dark_geometric_patchwork_dress/← 13 photos (Navy/maroon geometric crepe dress)
├── 08_blue_zigzag_chevron_dress/    ← 7 photos (Indigo blue chevron poly-crepe dress)
├── 09_gold_yellow_crinkle_dupatta/  ← 3 photos (Mustard gold crinkle metallic stole)
├── 10_pink_peach_embroidery_fabric/ ← 4 photos (Shimmer peach chanderi tissue fabric)
├── 11_dark_grey_textured_fabric/    ← 2 photos (Dark grey slub textured fabric piece)
├── 12_cream_beige_trousers/         ← 5 photos (Pleated cream PV formal trousers)
├── 13_red_casual_shirt/             ← 12 photos (Red poplin button-down + sprinkle shot)
├── 14_light_blue_formal_shirt/      ← 20 photos (Sky blue poly-cotton formal shirt)
├── 15_blue_white_check_shirt/       ← 10 photos (Blue & white check shirt + sprinkle shots)
├── 16_maroon_cream_floral_dress/    ← 14 photos (Maroon floral rayon house dress)
├── 17_brown_tan_pants/              ← 9 photos (Tan heavy cotton twill chinos)
├── 18_grey_white_plaid_shirt/       ← 7 photos (Grey & white block plaid shirt)
├── 19_beige_tan_shirt/              ← 7 photos (Beige slub linen casual shirt)
├── 20_yellow_floral_kurti/          ← 6 photos (Yellow rayon kurti with white flora)
├── 21_pink_orange_striped_kurti/    ← 8 photos (Pink & peach striped cotton kurti)
├── 22_peach_casual_shirt/           ← 22 photos (Peach broadcloth shirt full cycle)
├── 23_white_grid_check_shirt/       ← 12 photos (White windowpane check shirt)
├── 24_red_graphic_tshirt/           ← 9 photos (Red combed cotton jersey graphic tee)
├── 25_white_textured_shirt/         ← 10 photos (White micro-dobby textured formal shirt)
├── 26_dark_navy_shirt/              ← 10 photos (Dark navy solid button-down shirt)
├── 27_white_grey_mesh_shirt/        ← 4 photos (White/grey geometric mesh shirt)
├── 28_teal_embroidered_garment/     ← 1 photo (Teal georgette tunic with crochet lace)
├── 29_charcoal_formal_trousers_1/   ← 10 photos (Charcoal pleated PV formal pants)
├── 30_blue_grey_trousers/           ← 12 photos (Blue-grey chambray/denim trousers)
├── 31_dark_grey_formal_trousers_2/  ← 9 photos (Dark slate grey PV formal trousers)
├── 32_textured_formal_trousers_3/   ← 10 photos (Textured sharkskin PV formal trousers)
├── 33_white_grey_marble_shirt/      ← 1 photo (Marble print lightweight shirt)
├── 34_purple_floral_shirt/          ← 1 photo (Purple floral print lightweight shirt)
├── _batch_folded_stacks/            ← 4 photos (Close-up of finished folded piles)
├── _cart_setup_overview/            ← 6 photos (Environmental cart & iron box context)
├── DATASET_INDEX.md                 ← Technical file index
├── README.md                        ← Root GitHub README
└── rule_book_dk.md                  ← Master Project Rule Book & Living Reference
```

---

## Capture Environment & Physical Setup

### Physical Setup
- **Work Surface:** Traditional mobile four-wheeled wooden cart (*dhobi cart*) covered with a multi-layered padded wool/cotton cloth blanket and a top blue/striped ironing bed cover.
- **Heat & Pressure Tool:** Traditional heavy brass charcoal iron box (approx. 5–7 kg), containing burning coals for continuous high thermal inertia and sustained steam/heat pressure without electrical cords.
- **Fluid Application:** Manual hand-sprinkling of cold water drops to relax cellulose and synthetic fibers prior to heavy pressing.
- **Lighting Conditions:** Natural ambient outdoor sunlight with soft overhead canopy shade, creating distinct specular highlights on polished synthetic fabrics and deep contrast in cloth folds.

---

## Dataset Taxonomy & Multidimensional Variety

### 1. Apparel Categories (7 Classes)
1. **Women's Ethnic Kurtis & Dresses** (8 items | Folders `04, 05, 06, 07, 08, 16, 20, 21`)
2. **Women's Salwar Bottoms** (3 items | Folders `01, 02, 03`)
3. **Dupattas, Stoles & Fabrics** (3 items | Folders `09, 10, 11`)
4. **Men's Formal Shirts** (6 items | Folders `14, 15, 18, 23, 25, 26, 27`)
5. **Men's Casual Shirts** (6 items | Folders `13, 19, 22, 33, 34`)
6. **Men's Trousers & Chinos** (6 items | Folders `12, 17, 29, 30, 31, 32`)
7. **Graphic T-Shirts** (1 item | Folder `24`)

### 2. Fabric & Material Categories (10 Classes)
1. **Pure Cotton (Cambric / Plain Weave)** — folders `01, 02, 03, 21`
2. **Cotton Single-Jersey Knit** — folder `24`
3. **Rayon / Viscose Drape** — folders `06, 07, 16, 20`
4. **Slub Linen & Cotton Blends** — folders `05, 19`
5. **Poly-Cotton Shirting (Poplin / Broadcloth)** — folders `13, 14, 15, 18, 22`
6. **Micro-Dobby & Textured Weaves** — folders `23, 25, 27`
7. **Heavy Cotton Twill (Chinos)** — folder `17`
8. **Poly-Viscose (PV) Formal Suiting** — folders `12, 29, 31, 32`
9. **Chambray / Denim Weave** — folder `30`
10. **Crinkle Metallic Tissue & Georgette** — folders `09, 10, 28`

### 3. Visual Surface Patterns (8 Classes)
1. **Solid / Plain**: Folders `02, 03, 12, 13, 14, 17, 22, 26, 29, 31`
2. **Floral Prints**: Folders `06, 16, 20, 34`
3. **Embroidered (*Chikankari* / Lace / Resham)**: Folders `01, 05, 10, 28`
4. **Windowpane & Small Checks**: Folders `15, 23`
5. **Block Plaid**: Folder `18`
6. **Geometric & Patchwork**: Folders `07, 27`
7. **Chevron / Zigzag**: Folder `08`
8. **Vertical Stripes**: Folders `09, 21`

### 4. Procedural Ironing States (5 Visual States)
1. **Dry Wrinkled**: Baseline unpressed state.
2. **Water-Sprinkled**: Surface peppered with dark wet droplets.
3. **Active Hot Pressing**: Heat and physical pressure applied via the brass iron box.
4. **Crease Formation**: Precision mechanical fold-pressing (trouser pleats, collar edges).
5. **Folded & Stacked**: Final retail presentation fold and consolidated stack.

---

## Software & Tooling Stack

- **OS / Environment:** Windows PowerShell / Python 3.12.10
- **Computer Vision & Image Processing:** OpenCV (`cv2` 4.x), NumPy
- **Deep Learning & Modeling Readiness:** PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121, CUDA acceleration enabled
- **Version Control:** Git on `main` branch
- **Remote Host:** GitHub (`https://github.com/Dakshankarthic/iron.git`)

---

## Current Progress

### What's Working ✅

- [x] Workspace scanned, 335 items identified across timestamps 11:36 to 13:06
- [x] Time gap analysis and thumbnail contact sheets generated via OpenCV
- [x] Complete multimodal classification into 34 garment categories
- [x] Interleaved pressing and pre-iron water sprinkling correctly associated
- [x] Zero-loss local file move into structured folders
- [x] `DATASET_INDEX.md` and `README.md` generated
- [x] Git repository initialized, committed (`c2f618f`), and pushed to remote GitHub repo
- [x] Material diversity (10 classes) and visual taxonomy (8 patterns) quantified

### Main To-Do List

- [ ] **Generate Image Annotation Manifests:** Create a unified `annotations.json` or `metadata.csv` mapping each image path to its Garment ID, Category, Material, Color, Pattern, and State.
- [ ] **Train-Validation-Test Split:** Generate a reproducible train/val/test split (e.g. 70/15/15) preserving garment grouping to prevent data leakage.
- [ ] **Bounding Box & Garment Segmentation Masks:** Generate YOLO or COCO format labels for the garment area vs. the ironing table background.
- [ ] **Ironing State Classifier:** Benchmark a lightweight ResNet/ViT model to predict the 5 operational states (Wrinkled, Sprinkled, Pressing, Creased, Folded).
- [ ] **Dataset Card / Hugging Face Release:** Package dataset with standard Hugging Face `datasets` loader script.

---

## Known Issues & Solutions

| # | Problem | Solution | Status |
|---|---------|----------|--------|
| 1 | Raw dump of 335 files unorganized in workspace root | Structured into 34 numbered garment folders + 4 utility folders | ✅ Resolved |
| 2 | Camera sensor noise & blank pocket burst shots mixed with dataset | Filtered 5 blank/burst files into `_burst_and_noise_shots/` | ✅ Resolved |
| 3 | Android `.trashed-*` files present in directory | Safely isolated 13 deleted JPGs and 3 MP4 videos into `_trashed_items/` | ✅ Resolved |
| 4 | Pre-iron water sprinkling shots separated from final ironed garments | Mapped frames `099-109` to their exact corresponding shirts (`13, 15, 18, 19, 22, 23`) | ✅ Resolved |
| 5 | Interleaved pressing steps for peach shirt and pink kurti | Merged multi-step pressing photos into unified folders (`21` and `22`) | ✅ Resolved |
| 6 | Remote Git push size bottleneck (~1.17 GB) | Packaged clean root commit and completed uninterrupted push to GitHub `main` | ✅ Resolved |

---

## Build Phases

### Phase 1: Ingestion, Sorting, Cataloging & Remote Archival ✅ *(Finished: 2026-11-12)*
- [x] Analyze capture timestamps and image characteristics
- [x] Generate visual contact sheets
- [x] Mapped 319 active images into 34 garment folders
- [x] Create dataset documentation (`README.md`, `DATASET_INDEX.md`)
- [x] Initialize Git and push 1.17 GB to GitHub ([Dakshankarthic/iron](https://github.com/Dakshankarthic/iron))

### Phase 2: Metadata Extraction & Dataset Labeling *(Current Phase)*
- [ ] Generate structured `metadata.csv` / `dataset.json`
- [ ] Add EXIF metadata parser for focal length, exposure, and exact timestamps
- [ ] Annotate bounding boxes for garment region vs background ironing table

### Phase 3: Machine Learning & Vision Benchmarking
- [ ] Material classification benchmark (10 classes)
- [ ] Ironing state classification (5 states)
- [ ] Feature extraction using DINOv2 / CLIP embeddings

---

## Information Log (Chronological)

All milestones, inputs, and updates, timestamped for reference.

| Date & Time (IST) | Topic | Key Information |
|---|---|---|
| 2026-11-12 10:11 IST | Project Initiation & Constraints | User provided raw dataset in `I:\ironbox`. Specified that user will be away for 5 hours, requested zero permissions/uninterrupted autonomous workflow to organize every dress into its own folder. |
| 2026-11-12 10:21 IST | Departure & Duration Confirmation | User asked if they could leave now and how long it would take. Confirmed user can leave immediately, estimated completion under 2-3 minutes with zero permission required. |
| 2026-11-12 10:22 IST | Execution Approval | User confirmed "OKLAY". Full autonomous processing commenced. |
| 2026-11-12 10:26 IST | Processing & Reorganization Complete | Identified 335 items (319 active photos, 16 trashed/burst items). Created 34 garment folders and 4 utility folders. Generated `DATASET_INDEX.md` and detailed artifact report. Zero files lost. |
| 2026-11-12 18:37 IST | Methodology Explanation | User returned and asked "how do you do this". Explained timestamp gap analysis, OpenCV contact sheet generation, multimodal visual classification, and workflow reconstruction of the laundryman. |
| 2026-11-12 18:56 IST | Usable Count & Material Variety | User asked for total usable dataset size and variety in materials. Detailed 314 clean usable photos (304 garment + 10 context) across 10 distinct textile material classes. |
| 2026-11-12 20:08 IST | Full Catalog Request | User commanded "make the list now". Delivered detailed 4-group inventory covering all 34 garments with materials, colors, and captured stages. |
| 2026-11-12 20:22 IST | Total Dimension Variety Request | User asked for total types of variety. Detailed 4 major dimensions: 7 garment categories, 10 fabrics, 8 patterns, 11 color families, and 5 operational ironing states. |
| 2026-11-12 20:23 IST | GitHub Remote Push Request | User provided Git remote `https://github.com/Dakshankarthic/iron.git` and requested full push. |
| 2026-11-12 20:28 IST | Git Initialization & Commit | Initialized repository, copied `DATASET_INDEX.md` to `README.md`, staged all 337 files (1.17 GB), and created commit `c2f618f`. |
| 2026-11-12 20:36 IST | GitHub Remote Push Succeeded | `git push -u origin main` finished cleanly with code 0. Full dataset online on GitHub. |
| 2026-11-13 10:56 IST | Autonomous Execution Confirmed | User acknowledged autonomous execution ("you do it your self"). Outlined the 4 automated stages completed. |
| 2026-11-13 11:09 IST | Master Rule Book Request | User provided the Hexapod project `rule_book_dk.md` as reference template and requested a comprehensive master rule book for the Ironbox dataset project. |
| 2026-11-13 11:15 IST | Rule Book Authored | Created `I:\ironbox\rule_book_dk.md` with complete taxonomy, chronological log, AI rules, and continuation prompt. |
| 2026-11-14 19:10 IST | Noise & Trash Purge & Clean Readme | User requested removal of trashed videos/photos and noise burst images, and instructed to never write "Generated automatically by Antigravity" in README files. Permanently deleted `_trashed_items` (16 files) and `_burst_and_noise_shots` (5 files), leaving exactly 314 pure usable images across 36 folders. Removed generator footer from README.md and DATASET_INDEX.md, updated rule_book_dk.md, and pushed clean changes to GitHub remote. |

---

## 📐 Rules for AI Assistant

> **MANDATORY: Any AI assistant working on this project MUST follow these rules.**  
> **Read this entire README before writing any code or making any modifications.**

### Rule 1 — Read Before You Act
- **Always read this full `rule_book_dk.md`** at the start of every new chat session.
- Do NOT ask the user to re-explain things that are already documented here.
- Check the **Information Log**, **Dataset Taxonomy**, and **Build Phases** before starting work.

### Rule 2 — Log Everything with Timestamps
- **Every time the user provides new information or instructions**, add a new row to the **Information Log (Chronological)** table at the bottom of this document.
- Format: `| YYYY-MM-DD HH:MM IST | Topic | Key Information |`
- Use the user's local time (IST, UTC+5:30).
- This includes: new dataset annotations, scripts added, model training results, repository updates, and user design requests.

### Rule 3 — Update Progress as You Go
- When a task is completed, **check it off** in the **Current Progress** and **Build Phases** sections.
- When an entire Phase is completed, add the finished date to the header (e.g., `### Phase X ✅ *(Finished: YYYY-MM-DD)*`).
- When new tasks are identified, **add them** to the appropriate checklist.
- Update the `**Last Updated**` date at the top of the document, and move the old date into the `**Previous Updates**` list right below it. Never delete old update times.

### Rule 4 — Respect Confirmed Decisions
These decisions are FINAL. Do NOT alter folder structures or delete files without explicit confirmation:

| Decision | Value | Status |
|----------|-------|--------|
| Workspace Root | `I:\ironbox\` | ✅ Confirmed |
| Remote Git Repository | `https://github.com/Dakshankarthic/iron.git` | ✅ Confirmed |
| Git Branch | `main` | ✅ Confirmed |
| Core Garment Folders | 34 folders (`01_white_embroidered_salwar` to `34_purple_floral_shirt`) | ✅ Confirmed |
| Utility Folders | `_batch_folded_stacks`, `_cart_setup_overview` | ✅ Confirmed |
| File Integrity | Preserve all 314 clean usable photos; zero noise/trash items | ✅ Confirmed |
| Total Active Garments | 34 unique garments | ✅ Confirmed |
| Documentation Files | `README.md`, `DATASET_INDEX.md`, `rule_book_dk.md` | ✅ Confirmed |

### Rule 5 — Preserve File & Data Integrity
- Never delete or discard original photos. Any corrupted/noise frames must stay in `_burst_and_noise_shots`.
- Never modify original EXIF metadata or timestamps on source JPEG files.
- If generating derivatives (e.g., cropped bounding boxes, resized 640x640 images for YOLO, augmented copies), always save them into a new dedicated directory (e.g., `derivatives/` or `annotations/`), never inside the primary raw garment folders.

### Rule 6 — When User Gives New Info
When the user tells you something new about the project (new instructions, new models, new dataset tasks):
1. **Acknowledge** the info in your response.
2. **Add a row** to the Information Log table with the current date/time (IST).
3. **Update** any affected sections (progress checklists, to-do list, phase status).
4. **Update** the `Last Updated` timestamp at the top.

---

## Reference Links

- **GitHub Repository:** [https://github.com/Dakshankarthic/iron](https://github.com/Dakshankarthic/iron)
- **Local Index:** [DATASET_INDEX.md](file:///i:/ironbox/DATASET_INDEX.md)
- **Local README:** [README.md](file:///i:/ironbox/README.md)
- **PyTorch ImageFolder Docs:** [pytorch.org/vision/stable/datasets.html#imagefolder](https://pytorch.org/vision/stable/datasets.html#imagefolder)

---

## How to Continue in a New Chat Session

Copy and paste this into any new AI chat to provide full context:

> **Project:** 🧺 Ironbox Garment & Ironing Computer Vision Dataset  
> **Workspace:** `I:\ironbox`  
> **GitHub Repo:** `https://github.com/Dakshankarthic/iron.git` (Branch: `main`)  
> **Master Reference:** `I:\ironbox\rule_book_dk.md` — contains full dataset history, taxonomy, 34 garment categories, folder structure, AND rules for the AI assistant.  
> **Current Phase:** Phase 2 — Dataset Annotation, Metadata Extraction, and Vision Model Benchmarking.  
> **⚠️ Read the ENTIRE rule_book_dk.md first — especially the "Rules for AI Assistant" section — before writing any code or modifying the dataset.**

