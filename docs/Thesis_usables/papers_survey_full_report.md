# PDF Review Report — 35 Papers for Thesis Citation

Full report from the 2026-08-14 background paper-survey agent. See
`../../../../home/ws/udyyk/.claude/projects/-srv-THESIS/memory/thesis_papers_survey.md`
(the memory file) for the condensed, action-oriented version of this same material.

## Method note
All 35 PDFs were successfully read via full-text extraction (poppler/`pdftoppm` was not installed on this machine, so a `pypdf`-based extraction pipeline was built in a local venv — all 35 have genuine, machine-readable text layers, not scanned images; none were unreadable or corrupted). For each paper, title/author/abstract/intro pages were read; TABULA, CLIP, and RemoteCLIP were read more deeply as instructed, plus a couple of others that turned out to be unexpectedly central. A full-text, all-pages, case-insensitive search for "entrain" was run across all 35 files.

## ENTRAIN check — direct answer
**No ENTRAIN paper or ENTRAIN reference exists anywhere in this set of 35 PDFs.** The full-text search across every page of every file (CLIP+Lang folder and all 29 Features PDFs) returned zero matches for "entrain" in any form. TABULA, by contrast, is confirmed present and is exactly what its filename claims: `2013_IWU_LogaEtDiefenbach_TABULA-Calculation-Method.pdf`, "TABULA Calculation Method – Energy Use for Heating and Domestic Hot Water," Tobias Loga & Nikolaus Diefenbach, IWU, Jan 2013. If ENTRAIN is specifically wanted, it is not in this batch and needs to be sourced separately.

---

## (a) BORE discovery / stratified geospatial sampling methodology

**1-s2.0-S2210670723006856-main.pdf** — *"Data-driven classification of Urban Energy Units for district-level heating and electricity demand analysis,"* Luis Blanco, Alaa Alhamwi, Björn Schiricke, Bernhard Hoffschmidt (DLR/RWTH Aachen), *Sustainable Cities and Society* 101 (2024) 105075.
Classifies urban space into **16 "Urban Energy Units"** using open GIS data + a random forest (fills missing building attributes) followed by a decision tree (assigns the 16 classes), applied at scale to 8,249 units in Oldenburg, Germany, each carrying a typical energy-demand profile. Closest structural analog to BORE found in the set — cite it directly when justifying BORE's 16-class land-cover/settlement taxonomy and its use of a coarse classifier over open geodata to stratify a study area before per-unit feature/demand extraction.

**16.pdf** — *"Remote sensing of diverse urban environments: From the single city to multiple cities,"* Gang Chen, Yuyu Zhou, James A. Voogt, Eleanor C. Stokes, *Remote Sensing of Environment* 305 (2024) 114108.
Review paper arguing for a shift from single-city to multi-city (regional/global) urban remote sensing and documenting a systematic bias in the literature toward megacities and China/Europe/North America, with under-coverage of the Global South. Cite this to justify BORE's rationale for deliberately global, stratified (not megacity-biased) coordinate discovery — a direct literature-backed statement of the sampling-bias problem BORE is designed to avoid, including a concrete meta-analysis (79% of 644 papers reviewed 1980–2020 focused on a single city).

**8.pdf** *(secondary connection)* — Zhong et al., "A city-scale estimation of rooftop solar photovoltaic potential based on deep learning," *Applied Energy* 298 (2021) 117132 (MIT Senseable City Lab preprint). Introduces a "spatial optimization sampling strategy" using prior knowledge of urban/rural spatial layout and land use to cut labeled-sample collection cost by ~80% while improving model robustness across districts of different architectural styles. Supporting citation for the general principle that stratified/prior-informed sampling outperforms naive/random sampling for large-scale urban feature extraction — the same argument underlying BORE's tiered discovery design. Primary theme is (b), listed there too.

---

## (b) PORE multi-source feature extraction

Largest cluster — mostly solar-PV detection, rooftop/building extraction, and building-age estimation papers that map directly onto PORE's per-coordinate feature channels (solar potential, built-form, building age as an energy-relevant covariate).

**1.pdf** — Pena Pereira, Rafiee, Lhermitte, "Automated Rooftop Solar Panel Detection Through Convolutional Neural Networks," *Canadian Journal of Remote Sensing* 50:1 (2024). U-Net semantic segmentation of PV panels from 10cm aerial imagery (F1 up to 91.75%), studying how ground-truth characteristics (land use, rooftop color) affect model generalization across regions.

**2.pdf** — Ji et al. (DLR), "Solar photovoltaic module detection using laboratory and airborne imaging spectroscopy data," *Remote Sensing of Environment* 266 (2021). Physics-based (spectral-index) PV detection using hyperspectral data.

**3.pdf** — Li, Zhang, Guo et al., "Understanding rooftop PV panel semantic segmentation of satellite and aerial images for better using machine learning," *Advances in Applied Energy* 4 (2021). Characterizes PV imagery as class-imbalanced, non-concentrated, with a ~0.3 m resolution threshold for effective segmentation.

**4.pdf** — Jörges, Vidal, Hank, Bach, "Detection of Solar Photovoltaic Power Plants Using Satellite and Airborne Hyperspectral Imaging," *Remote Sensing* 15 (2023) 3403. First use of spaceborne PRISMA (30 m) hyperspectral data for PV detection.

**5.pdf** — Parhar, Sawasaki, Nusaputra, Vergara, Todeschini, Vahabi (UC Berkeley), "HyperionSolarNet: Solar Panel Detection from Aerial Images," NeurIPS 2021 Climate Change Workshop / arXiv 2201.02107. Two-branch (EfficientNet-B7 classifier + U-Net segmentation) pipeline for a global PV location/area map from a small labeled dataset.

**6.pdf** — Bouaziz, El Koundi, Ennine, "High-resolution solar panel detection in Sfax, Tunisia: A UNet-Based approach," *Renewable Energy* 235 (2024) 121171. Finds PV installations correlate with higher household income/well-maintained villas.

**7.pdf** — Jiang, Yao, Lu, Qin, Liu, Liu, Zhou, "Multi-resolution dataset for photovoltaic panel segmentation from satellite and aerial imagery," *Earth Syst. Sci. Data* 13 (2021). Open PV benchmark across three resolutions (0.8/0.3/0.1 m); cross-resolution model transfer fails without fine-tuning.

**8.pdf** — Zhong, Zhang, Chen et al., "A city-scale estimation of rooftop solar photovoltaic potential based on deep learning," *Applied Energy* 298 (2021) 117132. See note under theme (a); rooftop-extraction + solar-potential pipeline (330 km² rooftop area, 66 GW capacity estimated for Nanjing).

**9.pdf** — Zeng, Yang, Tang, Guan, Ma, "Intelligent Segmentation of Urban Building Roofs and Solar Energy Potential Estimation for Photovoltaic Applications," *J. Imaging* 11 (2025) 334. Couples rooftop segmentation (CESW-TransUNet) with physics-based shading/PVsyst simulation, showing 2D-area-only PV potential methods overestimate generation by 27.7%.

**10.pdf** — Dionelis et al. (ESA Φ-lab + partners), "Building Age Estimation: A New Multi-Modal Benchmark Dataset and Community Challenge (MapYourCity)," 2025. Multi-modal (VHR + Sentinel-2 + street-view) building-age classification benchmark across 19 European cities; top-view-only models still work reasonably when street-view is missing.

**11.pdf** — Sun, Zhang, Duarte (MIT Senseable City Lab), "Automatic Building Age Prediction from Street View Images," IC-NIDC 2021. DCNN on Amsterdam street-view images, 81% accuracy; motivated by sparse OSM building-age tag coverage.

**12.pdf** — Zeppelzauer, Despotovic, Sakeena, Koch, Döller, "Automatic Prediction of Building Age from Photographs," ICMR 2018. First patch-based CNN approach to building-age estimation from unconstrained photos; notes prior work attempted to visually derive heating energy demand directly from building photographs.

**17.pdf** — Zhao, Chen, Li, Ji, Sun, "Extracting Photovoltaic Panels From Heterogeneous Remote Sensing Images With Spatial and Spectral Differences," IEEE JSTARS 17 (2024). Fuses high-res Gaofen-2 with multispectral Sentinel-2 for PV segmentation (98% F1).

**18.pdf** — Jiang, Yao, Lu, Qin, Liu, Liu, Zhou, "Geospatial assessment of rooftop solar photovoltaic potential using multi-source remote sensing data," *Energy and AI* (2022). Combines geostationary-satellite solar-radiation inversion with building-footprint segmentation for hourly, 100 m-resolution PV potential across Jiangsu Province.

**19.pdf** — Zhuo, Huang, Liao, Tao, Zang, "BATSCCD: a new change detection method for mapping building age in rapidly changing urban areas using Landsat time series data," *Int. J. Digital Earth* (2024). Landsat time-series change detection for building age at yearly resolution.

**21.pdf** — Ma, Liu, Zhang, Tu, Zhou, Liu, Zheng, "Characterizing the Development of Photovoltaic Power Stations and Their Impacts on Vegetation Conditions from Landsat Time Series during 1990–2022," *Remote Sensing* 15 (2023) 3101. PV-station construction-year detection via Landsat/LandTrendr plus NDVI-based ecological impact.

**23.pdf** — Alkhatib, Al-Saad, Aburaed, Zitouni, Almansoori, Al-Ahmad, "Enhancing Photovoltaic Panel Segmentation in Remote Sensing Imagery: A Comparative Study of Attention-Integrated UNet Models," ISPRS Annals (2025). Benchmarks CBAM/SE/ECA/CA attention modules on UNet for PV segmentation on the PV01 UAV dataset.

**24.pdf** — Biljecki, Sindram, "Estimating Building Age with 3D GIS," ISPRS Annals IV-4/W5 (2017). Random-forest regression of building year-of-construction from 3D GIS attributes (height, footprint, storeys, neighbor age) in Rotterdam; best-case RMSE = 11 years. Explicitly states building age is "relied on heavily" for energy-demand estimation as an efficiency proxy.

**25.pdf** — Muhammed, Morsy, El-Shazly, "Building Rooftops Extraction for Solar PV Potential Estimation Using GIS-Based Methods," ISPRS Archives XLIV-M-3 (2021). SVM-based rooftop extraction (Egypt case study) feeding PVGIS/Solar Analyst tools.

**26.pdf** — Huang, Olson, Khalil, Saxe (U. Toronto), "Image-based prediction of residential building attributes with deep learning," *Journal of Industrial Ecology* 29 (2025). Predicts both floor area (regression, MAPE 19.4%) and age (classification, 70.3% acc.) from Google Street View using EfficientNetV2.

**28.pdf** — Starzyńska, Roussel, Jacoby, Asadipour (Royal College of Art), "Computer vision-based analysis of buildings and built environments: A systematic review of current approaches," arXiv 2208.00881 (2022). Systematic review of 88 CV-in-architecture papers across four clusters (landmark recognition, generative design, remote sensing, urban-environment analysis).

---

## (c) Heating/cooling demand estimation (degree-day methods, building energy calculation, TABULA/ENTRAIN-style)

**2013_IWU_LogaEtDiefenbach_TABULA-Calculation-Method.pdf** — see ENTRAIN check section above for full detail. **This is the central citation the author expects.** Full EN 13790-based seasonal-method specification: heating energy need = transmission + ventilation heat losses − (gain utilization factor × internal/solar gains), with DHW handled as a **separate, standardized per-m² net-energy-need constant** (10 kWh/m²·a single-unit, 15 kWh/m²·a multi-unit — not a `floor_area×11` single constant, but structurally the same idea). Cite eq. 1–3 (space heating) and eq. 16–18 + Table 1 (DHW standard values).

**13.pdf** — Anand, Deb, "The potential of remote sensing and GIS in urban building energy modelling," *Energy and Built Environment* 5 (2024) 957–969. 140-paper review mapping UBEM input variables (climate, geometry, construction, occupancy) to which remote-sensing/GIS techniques can supply them.

**20.pdf** — Mathur, Fennell, Rawal, Korolija, "Assessing a fit-for-purpose Urban Building Energy Modelling framework with reference to Ahmedabad," *Science and Technology for the Built Environment* (2021). "Level of Detail"/"Level of Effort" framework for UBEM inputs.

**22.pdf** — Mutani, Vocale, Javanroodi, "Toward Improved Urban Building Energy Modeling Using a Place-Based Approach," *Energies* 16 (2023) 3944. Formal three-way UBEM taxonomy (process-driven/data-driven/hybrid), tabulates physics-based UBEM tools (CitySim, TEASER, URBANopt).

**27.pdf** — Bishop, Gallardo, Williams, "A Review of Multi-Domain Urban Energy Modelling Data," *Clean Energy and Sustainability* 2 (2024) 10016. Review of UEM data across seven domains, flagging DHW consumption modeling and solar/wind DER data as under-reviewed sub-areas.

**29.pdf** — Kamel, "A Systematic Literature Review of Physics-Based Urban Building Energy Modeling (UBEM) Tools, Data Sources, and Challenges for Energy Conservation," *Energies* 15 (2022) 8649. PRISMA-based review of 88 physics-based UBEM case studies; lack of open high-res calibration data is the biggest adoption barrier.

**14.pdf** / **15.pdf** — see note below; both touch building-energy-consumption as one of five reviewed criteria but are broad multi-topic reviews, not degree-day method papers specifically.

---

## (d) CLIP-family vision-language pretraining / remote-sensing foundation models (EnergyCLIP background)

**CLIP.pdf** — Radford, Kim, Hallacy, Ramesh, Goh, Agarwal, Sastry, Askell, Mishkin, Clark, Krueger, Sutskever (OpenAI), "Learning Transferable Visual Models From Natural Language Supervision," ICML 2021 / arXiv 2103.00020. The foundational CLIP paper: contrastive image-text pretraining on 400M pairs, zero-shot transfer via natural-language class prompts.

**RemoteCLIP.pdf** — Fan Liu, Delong Chen, Zhangqingyun Guan, Xiaocong Zhou, Jiale Zhu, Qiaolin Ye, Liyong Fu, Jun Zhou, "RemoteCLIP: A Vision Language Foundation Model for Remote Sensing," IEEE TGRS. First domain-specific CLIP for remote sensing; solves pretraining-data scarcity via Box-to-Caption/Mask-to-Box conversion of heterogeneous RS annotations (12× larger corpus), beats CLIP by up to 6.39% zero-shot accuracy on 12 RS datasets.

**2602.20066v1.pdf** — Kundan Thota, Xuanhao Mu, Thorsten Schlachter, Veit Hagenmeyer (KIT), "HeatPrompt: Zero-Shot Vision-Language Modeling of Urban Heat Demand from Satellite Images," arXiv 2602.20066 (Feb 2026). **The single most directly on-point paper in the entire set.** Prompts a pretrained VLM as "an energy planner," extracts five salient visual heat-demand factors per satellite image + isoline mask, embeds the resulting captions (Nomic embeddings), regresses annual heat demand (93.7% R² uplift, 30% MAE reduction vs baseline).

**2603.17626v1.pdf** — Kundan Thota, Thorsten Schlachter, Veit Hagenmeyer (KIT), "A Multi-Agent System for Building-Age Cohort Mapping to Support Urban Energy Planning," arXiv 2603.17626 (Mar 2026). Three LLM agents (Zensus/OSM/Monument) fuse heterogeneous structured+unstructured sources into building-age cohorts, feeding a ConvNeXt+FPN satellite classifier ("BuildingAgeCNN," 90.69% accuracy).

**10.pdf** *(secondary)* — MapYourCity is also a multi-modal EO benchmark (not CLIP-based) but relevant background for "what multi-modal fusion in geospatial energy-adjacent ML currently looks like."

---

## (e) LLM-generated captions in geospatial/remote-sensing contexts

**2602.20066v1.pdf** (HeatPrompt) and **2603.17626v1.pdf** (Multi-Agent Building-Age) are the only two papers in the set that actually use LLM/VLM-generated text (captions, structured JSON extraction) as an intermediate representation for a downstream energy-relevant prediction. No other paper in the set performs LLM captioning; the remaining 33 are conventional CV/GIS/regression pipelines without a language-generation component.

---

## (f) Other / unclear / general background

**14.pdf** — Manapragada, Mandelmilch, Roitberg, Kizel, Natanian (Technion), "Remote sensing for environmentally responsive urban built environment: A review of tools, methods and gaps," *Remote Sensing Applications: Society and Environment* 38 (2025) 101529. Broad review (124 papers) across five environmental criteria: air quality, urban heat, thermal comfort, building energy, solar potential.

**15.pdf** — "Deepika," *IJCRT* 13(4) (April 2025). **Flag: near-verbatim identical in content/structure to paper 14 (same "124 research articles," same five criteria, same citations), attributed to a different single author in a different, lower-tier journal — looks like a derivative/unauthorized republication.** Recommend citing 14.pdf (the Elsevier original) only; treat 15.pdf as a duplicate to exclude.

**28.pdf** — also listed under (b); could sit here as general background/survey depending on chapter framing.

---

## Summary table

| File | Real title (short) | Primary theme(s) |
|------|---------------------|-------------------|
| 1-s2.0-S2210670723006856-main.pdf | Data-driven classification of Urban Energy Units | a, c |
| TABULA-Calculation-Method.pdf | TABULA Calculation Method (heating + DHW) | c (central) |
| 2602.20066v1.pdf | HeatPrompt (VLM heat demand) | d, c, e |
| 2603.17626v1.pdf | Multi-Agent Building-Age Cohort Mapping | b, e |
| CLIP.pdf | Learning Transferable Visual Models (CLIP) | d (central) |
| RemoteCLIP.pdf | RemoteCLIP | d (central) |
| Features/1.pdf | Rooftop Solar Panel Detection (CNN) | b |
| Features/2.pdf | Solar PV detection, imaging spectroscopy | b |
| Features/3.pdf | Rooftop PV semantic segmentation | b |
| Features/4.pdf | Solar PV Detection, Hyperspectral | b |
| Features/5.pdf | HyperionSolarNet | b |
| Features/6.pdf | Solar panel detection, Sfax Tunisia (UNet) | b |
| Features/7.pdf | Multi-resolution PV segmentation dataset | b |
| Features/8.pdf | City-scale rooftop solar PV potential | a, b |
| Features/9.pdf | Urban Building Roof Segmentation + PV potential | b |
| Features/10.pdf | Building Age Estimation (MapYourCity) | b, d |
| Features/11.pdf | Automatic Building Age from Street View | b |
| Features/12.pdf | Automatic Prediction of Building Age from Photos | b |
| Features/13.pdf | RS/GIS potential in UBEM (review) | c |
| Features/14.pdf | RS for environmentally responsive urban built env. (review) | c, b |
| Features/15.pdf | (near-duplicate of #14 — flagged, do not cite independently) | f |
| Features/16.pdf | Multi-city remote sensing (review) | a |
| Features/17.pdf | PV extraction, heterogeneous RS images | b |
| Features/18.pdf | Geospatial rooftop solar PV, multi-source RS | b, c |
| Features/19.pdf | BATSCCD building-age change detection | b, c |
| Features/20.pdf | Fit-for-purpose UBEM, Ahmedabad | c |
| Features/21.pdf | PV power stations + vegetation, Landsat | b |
| Features/22.pdf | Place-Based UBEM | c |
| Features/23.pdf | PV Segmentation, Attention UNet comparison | b |
| Features/24.pdf | Estimating Building Age with 3D GIS | b, c |
| Features/25.pdf | Rooftop Extraction for Solar PV (GIS) | b |
| Features/26.pdf | Image-based residential building attributes | b |
| Features/27.pdf | Multi-Domain Urban Energy Modelling Data (review) | c |
| Features/28.pdf | CV analysis of buildings (systematic review) | b, f |
| Features/29.pdf | Physics-Based UBEM Tools (systematic review) | c |
