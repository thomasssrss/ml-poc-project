const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title = "Prédiction du prix au m² à Paris — POC ML";

// ── Palette ──────────────────────────────────────────────────────────────────
const C = {
  black:   "1A1A1A",
  white:   "FFFFFF",
  bg:      "F9F9F9",
  accent:  "1B4F72",   // bleu marine foncé
  accent2: "2E86AB",   // bleu moyen (highlights)
  light:   "EBF3FB",   // bleu très clair (fond cards)
  muted:   "6B6B6B",
  border:  "E0E0E0",
  green:   "1A7A4A",
  red:     "B03A2E",
};

const FONT = "Calibri";
const FONT_TITLE = "Calibri";

// ── Helpers ──────────────────────────────────────────────────────────────────
function addTopBar(slide, accentColor = C.accent) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.07,
    fill: { color: accentColor }, line: { color: accentColor },
  });
}

function addSlideNumber(slide, num) {
  slide.addText(`${num}`, {
    x: 9.3, y: 5.3, w: 0.4, h: 0.2,
    fontSize: 9, color: C.muted, fontFace: FONT, align: "right",
  });
}

function addSectionLabel(slide, label) {
  slide.addText(label.toUpperCase(), {
    x: 0.45, y: 0.18, w: 9, h: 0.22,
    fontSize: 8, color: C.accent2, fontFace: FONT,
    bold: true, charSpacing: 3, margin: 0,
  });
}

function addTitle(slide, text, y = 0.5) {
  slide.addText(text, {
    x: 0.45, y, w: 9.1, h: 0.65,
    fontSize: 28, fontFace: FONT_TITLE, bold: true,
    color: C.black, margin: 0,
  });
}

function addSubtitle(slide, text, y = 1.15) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.45, y, w: 0.04, h: 0.28,
    fill: { color: C.accent2 }, line: { color: C.accent2 },
  });
  slide.addText(text, {
    x: 0.62, y, w: 8.9, h: 0.28,
    fontSize: 13, fontFace: FONT, color: C.accent, bold: true, margin: 0,
  });
}

function card(slide, x, y, w, h, color = C.light) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color },
    line: { color: C.border, width: 0.5 },
    shadow: { type: "outer", color: "000000", blur: 4, offset: 1, angle: 135, opacity: 0.07 },
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 1 — TITRE
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  // Fond divisé: gauche bleu marine, droite blanc
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 4.5, h: 5.625, fill: { color: C.accent }, line: { color: C.accent } });
  s.addShape(pres.shapes.RECTANGLE, { x: 4.5, y: 0, w: 5.5, h: 5.625, fill: { color: C.white }, line: { color: C.white } });

  // Titre côté gauche
  s.addText("Prédiction\ndu prix au m²\nà Paris", {
    x: 0.45, y: 1.2, w: 3.6, h: 2.8,
    fontSize: 30, fontFace: FONT_TITLE, bold: true,
    color: C.white, valign: "middle", margin: 0,
  });

  // Ligne décorative
  s.addShape(pres.shapes.RECTANGLE, { x: 0.45, y: 4.3, w: 1.2, h: 0.06, fill: { color: C.accent2 }, line: { color: C.accent2 } });

  // Sous-titre côté gauche
  s.addText("Proof of Concept — Machine Learning", {
    x: 0.45, y: 4.45, w: 3.7, h: 0.35,
    fontSize: 10, fontFace: FONT, color: "B0C8E0", margin: 0,
  });

  // Côté droit — détails
  s.addText("Données DVF", {
    x: 5.1, y: 1.5, w: 4.2, h: 0.4,
    fontSize: 11, fontFace: FONT, bold: true, color: C.accent, charSpacing: 2, margin: 0,
  });

  const stats = [
    ["156 077", "transactions analysées"],
    ["5", "datasets croisés"],
    ["30+", "features engineered"],
    ["1 412 €/m²", "meilleure MAE (Random Forest)"],
  ];
  stats.forEach(([num, label], i) => {
    const y = 2.1 + i * 0.78;
    card(s, 5.0, y, 4.5, 0.65, C.light);
    s.addText(num, { x: 5.15, y: y + 0.08, w: 1.6, h: 0.5, fontSize: 18, fontFace: FONT_TITLE, bold: true, color: C.accent, margin: 0 });
    s.addText(label, { x: 6.75, y: y + 0.15, w: 2.65, h: 0.35, fontSize: 10, fontFace: FONT, color: C.muted, margin: 0 });
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 2 — PROBLÉMATIQUE
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  addTopBar(s);
  addSectionLabel(s, "Contexte");
  addTitle(s, "Problématique & objectif");
  addSubtitle(s, "Comment estimer le prix au m² d'un appartement parisien ?");
  addSlideNumber(s, 2);

  // Carte centrale — question
  card(s, 0.45, 1.55, 9.1, 0.9, C.light);
  s.addText([
    { text: "Problème ML : ", options: { bold: true, color: C.accent } },
    { text: "Régression supervisée — prédire une valeur numérique continue", options: { color: C.black } },
  ], { x: 0.6, y: 1.65, w: 8.8, h: 0.55, fontSize: 13, fontFace: FONT, margin: 0 });

  // 3 cartes use-cases
  const cases = [
    { icon: "🏢", title: "Agence", body: "Proposer une première estimation rapide et objective" },
    { icon: "🏷️", title: "Vendeur", body: "Positionner son bien au juste prix marché" },
    { icon: "🔍", title: "Acheteur", body: "Détecter un bien surévalué ou une opportunité" },
  ];
  cases.forEach(({ icon, title, body }, i) => {
    const x = 0.45 + i * 3.07;
    card(s, x, 2.6, 2.85, 1.9, C.white);
    s.addShape(pres.shapes.RECTANGLE, { x, y: 2.6, w: 2.85, h: 0.07, fill: { color: C.accent2 }, line: { color: C.accent2 } });
    s.addText(icon, { x: x + 0.15, y: 2.72, w: 0.5, h: 0.5, fontSize: 22, margin: 0 });
    s.addText(title, { x: x + 0.15, y: 3.22, w: 2.55, h: 0.35, fontSize: 13, fontFace: FONT, bold: true, color: C.accent, margin: 0 });
    s.addText(body, { x: x + 0.15, y: 3.57, w: 2.55, h: 0.8, fontSize: 11, fontFace: FONT, color: C.muted, margin: 0 });
  });

  // Variable cible
  card(s, 0.45, 4.65, 9.1, 0.72, "F0F7FF");
  s.addText([
    { text: "Variable cible : ", options: { bold: true, color: C.accent } },
    { text: "prix_m² = valeur_foncière ÷ surface_réelle_bâtie", options: { color: C.black } },
  ], { x: 0.6, y: 4.75, w: 8.8, h: 0.5, fontSize: 12, fontFace: FONT, margin: 0 });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 3 — DONNÉES
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  addTopBar(s);
  addSectionLabel(s, "Dataset");
  addTitle(s, "5 datasets croisés");
  addSlideNumber(s, 3);

  const rows = [
    { icon: "📋", name: "DVF", detail: "156 077 ventes", source: "data.gouv.fr", apport: "Prix réels de transaction", highlight: true },
    { icon: "🚇", name: "Stations Métro/RER", detail: "405 stations", source: "IDFM open data", apport: "Proximité transports" },
    { icon: "🌳", name: "Espaces verts", detail: "2 528 espaces", source: "OpenData Paris", apport: "Proximité parcs & jardins" },
    { icon: "🏪", name: "BPE 2024", detail: "Équipements INSEE", source: "INSEE open data", apport: "Commerces, écoles, santé" },
    { icon: "🅿️", name: "Stationnement", detail: "24 874 places", source: "OpenData Paris", apport: "Accès voiture à Paris" },
  ];

  rows.forEach(({ icon, name, detail, source, apport, highlight }, i) => {
    const y = 1.2 + i * 0.82;
    const bg = highlight ? C.light : C.white;
    card(s, 0.45, y, 9.1, 0.73, bg);
    if (highlight) {
      s.addShape(pres.shapes.RECTANGLE, { x: 0.45, y, w: 0.06, h: 0.73, fill: { color: C.accent }, line: { color: C.accent } });
    }
    s.addText(icon, { x: 0.6, y: y + 0.13, w: 0.45, h: 0.45, fontSize: 18, margin: 0 });
    s.addText(name, { x: 1.1, y: y + 0.08, w: 2.2, h: 0.3, fontSize: 12, fontFace: FONT, bold: true, color: C.black, margin: 0 });
    s.addText(detail, { x: 1.1, y: y + 0.38, w: 2.2, h: 0.25, fontSize: 10, fontFace: FONT, color: C.muted, margin: 0 });
    s.addText(source, { x: 3.4, y: y + 0.2, w: 2.5, h: 0.3, fontSize: 10, fontFace: FONT, color: C.accent2, margin: 0 });
    s.addText(apport, { x: 5.95, y: y + 0.2, w: 3.4, h: 0.3, fontSize: 11, fontFace: FONT, color: C.black, margin: 0 });
  });

  s.addText("Méthode de jointure : BallTree haversine (distances sphériques vectorisées)", {
    x: 0.45, y: 5.25, w: 9.1, h: 0.25, fontSize: 9.5, fontFace: FONT,
    color: C.muted, italic: true, margin: 0,
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 4 — FEATURE ENGINEERING
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  addTopBar(s);
  addSectionLabel(s, "Préparation des données");
  addTitle(s, "Feature engineering");
  addSlideNumber(s, 4);

  // Colonne gauche — Nettoyage
  card(s, 0.45, 1.15, 4.3, 4.2, C.white);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.45, y: 1.15, w: 4.3, h: 0.06, fill: { color: C.accent }, line: { color: C.accent } });
  s.addText("Nettoyage DVF", { x: 0.6, y: 1.28, w: 4.0, h: 0.35, fontSize: 13, fontFace: FONT, bold: true, color: C.accent, margin: 0 });

  const cleaning = [
    "Filtrage : Vente / Paris / Appartement",
    "Surface > 0 (suppression lignes vides)",
    "Outliers retirés : prix_m² < 3 000 ou > 25 000 €",
    "188 426 → 156 077 lignes retenues",
  ];
  cleaning.forEach((txt, i) => {
    s.addText([{ text: txt }], {
      x: 0.6, y: 1.75 + i * 0.52, w: 4.0, h: 0.42,
      fontSize: 11, fontFace: FONT, color: C.black,
      bullet: true, margin: 0,
    });
  });

  // Funnel visuel
  const funnel = [["420 066", "fichier brut"], ["188 426", "après filtrage"], ["156 077", "après outliers"]];
  funnel.forEach(([n, l], i) => {
    const fy = 3.3 + i * 0.52;
    const fw = 3.6 - i * 0.4;
    const fx = 0.45 + (3.6 - fw) / 2 + 0.15;
    s.addShape(pres.shapes.RECTANGLE, { x: fx, y: fy, w: fw, h: 0.38, fill: { color: i === 2 ? C.accent : C.light }, line: { color: C.border } });
    s.addText(`${n} lignes — ${l}`, { x: fx, y: fy + 0.05, w: fw, h: 0.3, fontSize: 9.5, fontFace: FONT, color: i === 2 ? C.white : C.black, align: "center", margin: 0 });
  });

  // Colonne droite — Features
  card(s, 5.0, 1.15, 4.55, 4.2, C.white);
  s.addShape(pres.shapes.RECTANGLE, { x: 5.0, y: 1.15, w: 4.55, h: 0.06, fill: { color: C.accent2 }, line: { color: C.accent2 } });
  s.addText("Features créées", { x: 5.15, y: 1.28, w: 4.2, h: 0.35, fontSize: 13, fontFace: FONT, bold: true, color: C.accent, margin: 0 });

  const features = [
    ["log_surface", "Relation non-linéaire surface/prix"],
    ["surface_par_piece", "Proxy du standing du bien"],
    ["arrondissement_prix_moyen", "Target encoding (train set)"],
    ["distance_station + nb_stations_500m", "6 features transport"],
    ["distance_ev + nb_ev_500m", "3 features espaces verts"],
    ["distance_* BPE + nb_*_500m", "8 features équipements"],
    ["distance_parking + nb_parking_300m", "2 features stationnement"],
  ];
  features.forEach(([feat, desc], i) => {
    const fy = 1.75 + i * 0.49;
    s.addText(feat, { x: 5.15, y: fy, w: 2.2, h: 0.28, fontSize: 10, fontFace: FONT, bold: true, color: C.accent2, margin: 0 });
    s.addText(desc, { x: 5.15, y: fy + 0.24, w: 4.2, h: 0.22, fontSize: 9.5, fontFace: FONT, color: C.muted, margin: 0 });
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 5 — MODÈLES
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  addTopBar(s);
  addSectionLabel(s, "Modélisation");
  addTitle(s, "3 modèles comparés");
  addSlideNumber(s, 5);

  const models = [
    {
      num: "01", name: "Régression Linéaire", tag: "Baseline",
      color: C.muted,
      params: ["Pipeline avec RobustScaler", "Modèle de référence interprétable", "Hypothèse de linéarité"],
    },
    {
      num: "02", name: "Random Forest", tag: "Meilleur modèle",
      color: C.accent,
      params: ["200 arbres de décision", "max_depth = 20", "min_samples_leaf = 5", "Capture les non-linéarités"],
    },
    {
      num: "03", name: "Gradient Boosting", tag: "HistGBT (sklearn)",
      color: C.accent2,
      params: ["500 itérations", "learning_rate = 0.05", "Boosting séquentiel"],
    },
  ];

  models.forEach(({ num, name, tag, color, params }, i) => {
    const x = 0.45 + i * 3.2;
    card(s, x, 1.1, 3.0, 3.8, C.white);
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.1, w: 3.0, h: 0.08, fill: { color }, line: { color } });
    s.addText(num, { x: x + 0.15, y: 1.28, w: 0.5, h: 0.4, fontSize: 22, fontFace: FONT_TITLE, bold: true, color, margin: 0 });
    s.addText(tag, { x: x + 0.15, y: 1.72, w: 2.7, h: 0.28, fontSize: 9, fontFace: FONT, color, bold: true, charSpacing: 1.5, margin: 0 });
    s.addText(name, { x: x + 0.15, y: 1.98, w: 2.7, h: 0.42, fontSize: 13, fontFace: FONT, bold: true, color: C.black, margin: 0 });
    params.forEach((p, j) => {
      s.addText([{ text: p }], { x: x + 0.15, y: 2.55 + j * 0.37, w: 2.7, h: 0.32, fontSize: 10.5, fontFace: FONT, color: C.black, bullet: true, margin: 0 });
    });
  });

  // Protocole
  card(s, 0.45, 5.05, 9.1, 0.38, C.light);
  s.addText([
    { text: "Protocole : ", options: { bold: true, color: C.accent } },
    { text: "Split 80% train / 20% test  ·  Métriques : MAE, RMSE, R²  ·  Log-transform de la cible testé en v2", options: { color: C.black } },
  ], { x: 0.6, y: 5.1, w: 8.8, h: 0.28, fontSize: 10.5, fontFace: FONT, margin: 0 });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 6 — RÉSULTATS
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  addTopBar(s);
  addSectionLabel(s, "Résultats");
  addTitle(s, "Performances des modèles");
  addSlideNumber(s, 6);

  // Tableau
  const headerRow = [
    { text: "Modèle", options: { bold: true, color: C.white, fill: { color: C.accent }, fontSize: 11 } },
    { text: "MAE (€/m²)", options: { bold: true, color: C.white, fill: { color: C.accent }, fontSize: 11, align: "center" } },
    { text: "RMSE (€/m²)", options: { bold: true, color: C.white, fill: { color: C.accent }, fontSize: 11, align: "center" } },
    { text: "R²", options: { bold: true, color: C.white, fill: { color: C.accent }, fontSize: 11, align: "center" } },
  ];
  const data = [
    ["Régression Linéaire", "2 026", "2 951", "0.236"],
    ["Random Forest v1", "1 572", "2 357", "0.513"],
    ["Gradient Boosting v1", "1 825", "2 690", "0.365"],
    ["Random Forest v2 (log) ← MEILLEUR", "1 412", "2 227", "0.565"],
    ["Gradient Boosting v2 (log)", "1 826", "2 739", "0.342"],
  ];

  const tableRows = [headerRow, ...data.map((row, i) => {
    const isBest = i === 3;
    return row.map((cell, j) => ({
      text: cell,
      options: {
        fontSize: 11,
        fontFace: FONT,
        color: isBest ? (j === 0 ? C.accent : C.green) : C.black,
        bold: isBest,
        fill: { color: isBest ? C.light : (i % 2 === 0 ? C.white : "FAFAFA") },
        align: j === 0 ? "left" : "center",
      },
    }));
  })];

  s.addTable(tableRows, {
    x: 0.45, y: 1.1, w: 9.1,
    colW: [4.1, 1.6, 1.8, 1.6],
    border: { pt: 0.5, color: C.border },
    rowH: 0.48,
  });

  // Stats clés
  const kpis = [
    { val: "1 412 €/m²", label: "Meilleure MAE", sub: "Random Forest v2" },
    { val: "0.565", label: "Meilleur R²", sub: "56.5% de la variance expliquée" },
    { val: "~14%", label: "Erreur relative", sub: "Sur prix médian 10 300 €/m²" },
  ];
  kpis.forEach(({ val, label, sub }, i) => {
    const x = 0.45 + i * 3.07;
    card(s, x, 4.08, 2.85, 1.3, i === 0 ? C.light : C.white);
    if (i === 0) s.addShape(pres.shapes.RECTANGLE, { x, y: 4.08, w: 2.85, h: 0.06, fill: { color: C.accent }, line: { color: C.accent } });
    s.addText(val, { x: x + 0.15, y: 4.18, w: 2.55, h: 0.52, fontSize: 22, fontFace: FONT_TITLE, bold: true, color: i === 0 ? C.accent : C.black, margin: 0 });
    s.addText(label, { x: x + 0.15, y: 4.72, w: 2.55, h: 0.28, fontSize: 11, fontFace: FONT, bold: true, color: C.black, margin: 0 });
    s.addText(sub, { x: x + 0.15, y: 4.98, w: 2.55, h: 0.28, fontSize: 9.5, fontFace: FONT, color: C.muted, margin: 0 });
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 7 — VISUALISATIONS / EDA
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  addTopBar(s);
  addSectionLabel(s, "Analyse exploratoire");
  addTitle(s, "Principaux résultats EDA");
  addSlideNumber(s, 7);

  // Bar chart — prix par arrondissement (extrait)
  const arrData = [
    { arr: "6e", prix: 14646 },
    { arr: "7e", prix: 14483 },
    { arr: "4e", prix: 12933 },
    { arr: "8e", prix: 12304 },
    { arr: "15e", prix: 10300 },
    { arr: "18e", prix: 9321 },
    { arr: "20e", prix: 8670 },
    { arr: "19e", prix: 8409 },
  ];
  s.addChart(pres.charts.BAR, [{
    name: "Prix médian",
    labels: arrData.map(d => d.arr),
    values: arrData.map(d => d.prix),
  }], {
    x: 0.45, y: 1.05, w: 5.7, h: 3.5,
    barDir: "bar",
    chartColors: [C.accent2, C.accent2, C.accent2, C.accent2, "8E8E8E", C.muted, C.muted, C.muted],
    chartArea: { fill: { color: C.white } },
    catAxisLabelColor: C.black,
    valAxisLabelColor: C.muted,
    valGridLine: { color: C.border, size: 0.5 },
    catGridLine: { style: "none" },
    showValue: true,
    dataLabelColor: C.black,
    dataLabelFontSize: 9,
    showLegend: false,
    title: "Prix médian €/m² par arrondissement",
    showTitle: true,
    titleFontSize: 11,
    titleColor: C.black,
  });

  // 3 insights
  const insights = [
    { icon: "📍", title: "Écart max–min", body: "6e (14 646 €/m²) vs 19e (8 409 €/m²) → écart de 6 237 €/m²" },
    { icon: "🔑", title: "Variable clé", body: "L'arrondissement est la variable la plus discriminante du modèle" },
    { icon: "📈", title: "Log-transform", body: "Améliore la MAE de 1 572 → 1 412 €/m² (+10% de précision)" },
  ];
  insights.forEach(({ icon, title, body }, i) => {
    const y = 1.1 + i * 1.42;
    card(s, 6.3, y, 3.3, 1.25, C.white);
    s.addShape(pres.shapes.RECTANGLE, { x: 6.3, y, w: 0.05, h: 1.25, fill: { color: C.accent2 }, line: { color: C.accent2 } });
    s.addText(icon, { x: 6.42, y: y + 0.08, w: 0.42, h: 0.42, fontSize: 18, margin: 0 });
    s.addText(title, { x: 6.42, y: y + 0.52, w: 3.0, h: 0.28, fontSize: 11, fontFace: FONT, bold: true, color: C.accent, margin: 0 });
    s.addText(body, { x: 6.42, y: y + 0.78, w: 3.0, h: 0.4, fontSize: 10, fontFace: FONT, color: C.black, margin: 0 });
  });

  s.addText("Prix médian Paris : 10 300 €/m²", {
    x: 0.45, y: 4.65, w: 5.7, h: 0.28, fontSize: 10, fontFace: FONT,
    color: C.muted, italic: true, align: "center", margin: 0,
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 8 — APPLICATION
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  addTopBar(s);
  addSectionLabel(s, "Application");
  addTitle(s, "Application Streamlit");
  addSlideNumber(s, 8);

  // 5 pages — liste horizontale
  const pages = ["Le Projet", "Données brutes", "Feature Engineering", "Performances", "Estimer un prix"];
  pages.forEach((p, i) => {
    const x = 0.45 + i * 1.83;
    const isLast = i === 4;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.1, w: 1.7, h: 0.48,
      fill: { color: isLast ? C.accent : C.light },
      line: { color: isLast ? C.accent : C.border },
    });
    s.addText(p, { x, y: 1.16, w: 1.7, h: 0.36, fontSize: 9.5, fontFace: FONT, bold: isLast, color: isLast ? C.white : C.black, align: "center", margin: 0 });
  });

  // 2 modes estimateur
  const modes = [
    {
      title: "Par arrondissement", tag: "Rapide",
      color: C.muted,
      items: ["Sélectionner l'arrondissement", "Surface + nombre de pièces", "Étage + présence d'ascenseur", "Résultat immédiat"],
    },
    {
      title: "Par adresse exacte", tag: "Précis",
      color: C.accent,
      items: [
        "Géocodage via api-adresse.data.gouv.fr",
        "Prix IDW sur 20 arrondissements",
        "Bonus/malus type de voie (avenue +5%, impasse -5%)",
        "Ajustements : métro, parcs, commerces, stationnement",
        "Ajustement étage + sans ascenseur",
      ],
    },
  ];
  modes.forEach(({ title, tag, color, items }, i) => {
    const x = 0.45 + i * 4.65;
    const w = i === 0 ? 4.3 : 5.1;
    card(s, x, 1.75, w, 3.65, C.white);
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.75, w, h: 0.07, fill: { color }, line: { color } });
    s.addText(tag, { x: x + 0.15, y: 1.9, w: w - 0.3, h: 0.28, fontSize: 9, fontFace: FONT, bold: true, color, charSpacing: 2, margin: 0 });
    s.addText(title, { x: x + 0.15, y: 2.2, w: w - 0.3, h: 0.38, fontSize: 14, fontFace: FONT, bold: true, color: C.black, margin: 0 });
    items.forEach((item, j) => {
      s.addText([{ text: item }], { x: x + 0.15, y: 2.7 + j * 0.42, w: w - 0.3, h: 0.38, fontSize: 10.5, fontFace: FONT, color: C.black, bullet: true, margin: 0 });
    });
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 9 — LIMITES & PISTES
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  addTopBar(s);
  addSectionLabel(s, "Discussion");
  addTitle(s, "Limites & pistes d'amélioration");
  addSlideNumber(s, 9);

  // Gauche — Limites
  card(s, 0.45, 1.1, 4.3, 4.25, C.white);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.45, y: 1.1, w: 4.3, h: 0.07, fill: { color: C.red }, line: { color: C.red } });
  s.addText("Limites actuelles", { x: 0.6, y: 1.24, w: 4.0, h: 0.35, fontSize: 13, fontFace: FONT, bold: true, color: C.red, margin: 0 });
  const limits = [
    "R² = 0.565 → 43% de variance inexpliquée",
    "DVF sans : étage, état du bien, balcon, exposition",
    "Coordonnées GPS approximatives (centroïde cadastral)",
    "Données externes = état actuel ≠ moment des ventes",
  ];
  limits.forEach((l, i) => {
    s.addText([{ text: l }], { x: 0.6, y: 1.72 + i * 0.62, w: 4.0, h: 0.52, fontSize: 11, fontFace: FONT, color: C.black, bullet: true, margin: 0 });
  });

  // Droite — Pistes
  card(s, 5.0, 1.1, 4.55, 4.25, C.white);
  s.addShape(pres.shapes.RECTANGLE, { x: 5.0, y: 1.1, w: 4.55, h: 0.07, fill: { color: C.green }, line: { color: C.green } });
  s.addText("Pistes d'amélioration", { x: 5.15, y: 1.24, w: 4.2, h: 0.35, fontSize: 13, fontFace: FONT, bold: true, color: C.green, margin: 0 });
  const improvements = [
    "Enrichir avec données DPE (classe énergie)",
    "Features qualitatives via annonces immobilières",
    "Modèle géographique (krigeage ou GNN)",
    "Mise à jour automatique DVF (nouvelles ventes)",
    "XGBoost / LightGBM pour de meilleures performances",
  ];
  improvements.forEach((p, i) => {
    s.addText([{ text: p }], { x: 5.15, y: 1.72 + i * 0.62, w: 4.2, h: 0.52, fontSize: 11, fontFace: FONT, color: C.black, bullet: true, margin: 0 });
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 10 — CONCLUSION (fond sombre)
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 5.625, fill: { color: C.accent }, line: { color: C.accent } });

  s.addText("Conclusion", {
    x: 0.6, y: 0.5, w: 8.8, h: 0.7,
    fontSize: 32, fontFace: FONT_TITLE, bold: true, color: C.white, margin: 0,
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.22, w: 1.4, h: 0.055, fill: { color: C.accent2 }, line: { color: C.accent2 } });

  const conclusions = [
    { icon: "✓", text: "POC fonctionnel de bout en bout : données → modèle → application" },
    { icon: "✓", text: "Random Forest + log-transform : meilleur compromis performance / interprétabilité" },
    { icon: "✓", text: "MAE = 1 412 €/m² sur 156 077 transactions parisiennes" },
    { icon: "✓", text: "Estimateur temps réel par adresse exacte (géocodage + IDW + proximités)" },
    { icon: "✓", text: "Données 100% open data — projet entièrement reproductible" },
  ];
  conclusions.forEach(({ icon, text }, i) => {
    card(s, 0.6, 1.4 + i * 0.74, 8.8, 0.63, "234F6E");
    s.addText(icon, { x: 0.75, y: 1.48 + i * 0.74, w: 0.35, h: 0.42, fontSize: 14, color: C.accent2, margin: 0 });
    s.addText(text, { x: 1.15, y: 1.48 + i * 0.74, w: 8.1, h: 0.42, fontSize: 12, fontFace: FONT, color: C.white, valign: "middle", margin: 0 });
  });

  s.addText("Merci", { x: 7.5, y: 5.1, w: 2.1, h: 0.38, fontSize: 13, fontFace: FONT, color: "7AAFC8", italic: true, align: "right", margin: 0 });
}

// ── Écriture du fichier ──────────────────────────────────────────────────────
pres.writeFile({ fileName: "/Users/thomasrousseau/Documents/ml-poc-project/deliverables/soutenance.pptx" })
  .then(() => console.log("✅ soutenance.pptx créé"))
  .catch(e => console.error("❌", e));
