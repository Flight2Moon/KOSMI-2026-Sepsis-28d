'use strict';

// Read the same published estimates that remain accessible in the numerical table.
const get = id => document.getElementById(id);
const modelFilter = get('model-filter');
const cohortFilter = get('cohort-filter');
const models = Array.from(modelFilter.options).slice(1).map(option => option.value);
const cohorts = Array.from(cohortFilter.options).slice(1).map(option => option.value);
const cohortHeadings = ['MIMIC-IV', 'eICU alt', 'eICU primary', 'SICdb', 'NWICU'];
const records = Array.from(get('all-data').tBodies[0].rows, row => {
  const [cohort, model, n, flat, two, delta, interval] = row.cells;
  const low = Number(interval.dataset.low);
  const high = Number(interval.dataset.high);
  return {
    cohort: cohort.textContent, model: model.textContent, n: n.textContent, row,
    flat: flat.dataset, two: two.dataset, delta: Number(delta.dataset.value), low, high,
    result: low > 0 ? 'multi' : high < 0 ? 'two' : 'neutral'
  };
});
const symbols = {multi: '●', two: '◆', neutral: '○'};
const conclusions = {
  multi: 'Multiclass higher at nominal significance',
  two: 'Two-stage higher at nominal significance',
  neutral: 'The interval includes zero'
};
const matchesView = {
  all: () => true,
  small: r => Math.abs(r.delta) < 0.02,
  large: r => Math.abs(r.delta) >= 0.02,
  multi: r => r.result === 'multi',
  two: r => r.result === 'two',
  neutral: r => r.result === 'neutral'
};
const controls = Array.from(document.querySelectorAll('[data-view]'));
const params = new URLSearchParams(location.search);
let view = Object.hasOwn(matchesView, params.get('view')) ? params.get('view') : 'all';
modelFilter.value = models.includes(params.get('model')) ? params.get('model') : '';
cohortFilter.value = cohorts.includes(params.get('cohort')) ? params.get('cohort') : '';
let selected = records.find(r => r.model === 'XGBoost' && r.cohort === 'MIMIC-IV Temporal');
let visible = [];
const cellButtons = new Map();
const forestButtons = new Map();
const fixed = v => Number(v).toFixed(4);
const signed = v => (v < 0 ? '−' : '+') + fixed(Math.abs(v));
const element = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
};
function intervalText(r) {
  return signed(r.low) + ' to ' + signed(r.high);
}
function comparisonLabel(r) {
  return r.model + ', ' + r.cohort + ', difference ' + signed(r.delta) + ', ' + conclusions[r.result];
}
function headerCell(label, scope) {
  const th = element('th', label);
  th.scope = scope;
  return th;
}
function selectRecord(record) {
  selected = record;
  for (const buttons of [cellButtons, forestButtons]) {
    for (const [r, button] of buttons) {
      button.classList.toggle('active', r === record);
      button.setAttribute('aria-pressed', String(r === record));
    }
  }
  const detail = get('comparison-detail');
  if (!record) {
    detail.replaceChildren(element('h3', 'No matching comparisons'),
      element('p', 'Choose another finding, algorithm, or cohort. Reset filters to see all 25.'));
    return;
  }
  const estimates = element('dl');
  for (const [label, score] of [['Direct multiclass', record.flat], ['Death-first two-stage', record.two]]) {
    estimates.append(element('dt', label), element('dd',
      fixed(score.value) + ' (95% CI ' + fixed(score.low) + ' to ' + fixed(score.high) + ')'));
  }
  detail.replaceChildren(element('p', 'Selected comparison', 'kicker'), element('h3', record.model),
    element('span', record.cohort + ' · ' + record.n + ' patients', 'cohort'),
    element('strong', 'Δ ' + signed(record.delta), 'delta'),
    element('span', '95% CI ' + intervalText(record), 'bounds'), estimates,
    element('p', conclusions[record.result], 'conclusion'));
}
function buildMatrix() {
  cellButtons.clear();
  const matrix = element('table', undefined, 'grid-table');
  matrix.append(element('caption', 'Difference in macro AUROC by algorithm and evaluation cohort', 'sr'));
  const chosenSites = cohortFilter.value ? [cohortFilter.value] : cohorts;
  const chosenModels = modelFilter.value ? [modelFilter.value] : models;
  const head = matrix.createTHead().insertRow();
  head.append(headerCell('Algorithm', 'col'));
  for (const site of chosenSites) {
    const th = headerCell(cohortHeadings[cohorts.indexOf(site)], 'col');
    th.title = site;
    head.append(th);
  }
  const body = matrix.createTBody();
  for (const model of chosenModels) {
    const row = body.insertRow();
    row.append(headerCell(model, 'row'));
    for (const site of chosenSites) {
      const td = row.insertCell();
      const r = visible.find(entry => entry.cohort === site && entry.model === model);
      if (!r) {
        td.append(element('span', '–', 'grid-empty'));
        td.setAttribute('aria-label', 'Outside this filter');
        continue;
      }
      const button = element('button', undefined, 'cell ' + r.result);
      button.type = 'button';
      button.setAttribute('aria-label', comparisonLabel(r));
      button.title = comparisonLabel(r);
      button.append(element('span', symbols[r.result], 'symbol'), element('span', signed(r.delta), 'estimate'));
      button.addEventListener('click', () => {
        selectRecord(r);
        if (matchMedia('(max-width: 900px)').matches) get('comparison-detail').scrollIntoView({block: 'nearest'});
      });
      cellButtons.set(r, button);
      td.append(button);
    }
  }
  get('matrix').replaceChildren(matrix);
}
// A common horizontal scale keeps interval lengths comparable across all filters.
const position = x => (x + 0.06) * 1000;
function drawForest() {
  forestButtons.clear();
  const forest = get('forest');
  if (!visible.length) {
    forest.replaceChildren(element('p', 'No comparisons match these filters.'));
    return;
  }
  const axis = element('div', undefined, 'forest-axis');
  const ticks = element('div', undefined, 'axis');
  ticks.append(element('span', '−0.06'), element('span','0','zero'), element('span','+0.04'));
  axis.append(element('span','Algorithm / cohort'), ticks, element('span','Δ (95% CI)'));
  forest.replaceChildren(axis);
  for (const r of visible) {
    const button = element('button', undefined, 'forest-row');
    button.type = 'button';
    button.setAttribute('aria-label', comparisonLabel(r) + ', confidence interval ' + intervalText(r));
    const track = element('span', undefined, 'interval ' + r.result);
    const bar = element('i', undefined, 'bar');
    bar.style.left = position(r.low) + '%';
    bar.style.width = (position(r.high) - position(r.low)) + '%';
    const point = element('i', undefined, 'point');
    point.style.left = position(r.delta) + '%';
    track.append(bar, point);
    button.append(element('span', r.model + ' · ' + r.cohort), track,
      element('span', signed(r.delta) + ' (' + intervalText(r) + ')'));
    button.addEventListener('click', () => {
      selectRecord(r);
      get('comparison-detail').scrollIntoView({block: 'nearest'});
    });
    forestButtons.set(r, button);
    forest.append(button);
  }
}
function render() {
  visible = records.filter(r => matchesView[view](r) &&
    (!modelFilter.value || r.model === modelFilter.value) &&
    (!cohortFilter.value || r.cohort === cohortFilter.value));
  for (const r of records) r.row.hidden = !visible.includes(r);
  for (const btn of controls) btn.setAttribute('aria-pressed', String(btn.dataset.view === view));
  get('filtered-count').textContent = visible.length + ' of 25 comparisons shown';
  buildMatrix();
  drawForest();
  selectRecord(visible.includes(selected) ? selected : visible[0]);
}
function update() {
  const url = new URL(location.href);
  for (const [key, value] of [['view', view === 'all' ? '' : view], ['model', modelFilter.value], ['cohort', cohortFilter.value]]) {
    if (value) url.searchParams.set(key, value);
    else url.searchParams.delete(key);
  }
  history.replaceState(null, '', url);
  render();
}
function applyView(value) {
  view = value;
  modelFilter.value = '';
  cohortFilter.value = '';
  update();
}
for (const btn of controls) btn.addEventListener('click', () => applyView(btn.dataset.view));
get('reset').addEventListener('click', () => applyView('all'));
for (const filter of [modelFilter, cohortFilter]) filter.addEventListener('change', update);
render();
get('explorer').hidden = false;
