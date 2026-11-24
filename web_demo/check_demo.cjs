// Check the browser demo against the Python reference values in manifest.json.
//   node web_demo/check_demo.cjs [demo directory]
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');

const dir = path.resolve(process.argv[2] || __dirname);
const read = f => fs.readFileSync(path.join(dir, f));
const sha = f => crypto.createHash('sha256').update(read(f)).digest('hex');
const manifest = JSON.parse(read('manifest.json'));
assert.equal(sha('data.json'), manifest.data_sha256, 'data.json hash');
for (const [f, h] of Object.entries(manifest.files_sha256)) assert.equal(sha(f), h, `${f} hash`);

const data = JSON.parse(read('data.json'));
const { meta, items, evenings } = data;
assert.equal(items.length, manifest.items);
assert.equal(evenings.length, manifest.evenings);
for (const ev of evenings) {
  for (const key of ['on_hand', 'pipeline', 'sigma']) assert.equal(ev[key].length, items.length);
  assert.equal(ev.forecast.length, items.length);
  assert(ev.forecast.every(row => row.length === meta.horizon && row.every(v => v >= 0)));
}

const model = require(path.join(dir, 'model.js'));
const close = (a, b, rel, what) => assert(Math.abs(a - b) <= rel * Math.max(1, Math.abs(b)), `${what}: ${a} vs ${b}`);

require(path.join(dir, 'vendor/highs.js'))({ locateFile: f => path.join(dir, 'vendor', f) }).then(highs => {
  const avg = meta.avg_daily_total;
  const K = meta.base_delivery_mult * avg, C = meta.base_storage_days * avg;
  const gaps = [];
  for (const c of manifest.checks) {
    const ev = evenings[c.evening];
    assert.equal(ev.date, c.date);
    const res = model.planLP(highs, ev, items, meta, K, C);
    close(res.objective, c.lp_objective, 1e-7, `${c.date} LP objective`);
    const tight = model.planLP(highs, ev, items, meta, 0.95 * avg, 2.5 * avg);
    close(tight.objective, c.lp_objective_tight, 1e-7, `${c.date} LP objective, tight limits`);
    const tonight = res.orders.reduce((s, row) => s + row[0], 0);
    assert(tonight <= K + 1e-6, 'delivery limit');
    const rule = model.ruleOrders(ev, meta, meta.rule_z, K, C);
    rule.forEach((v, i) => assert(Math.abs(v - c.rule_tonight[i]) <= 1e-3 + 1e-9, `${c.date} rule order ${i}`));
    const rr = model.replay(evenings, c.evening, 7, items, meta, C, e => model.ruleOrders(e, meta, meta.rule_z, K, C)).totals;
    for (const k of ['demand', 'lost_units', 'lost_usd', 'holding_usd', 'returned_units', 'total_usd']) close(rr[k], c.rule_replay[k], 1e-9, `${c.date} rule replay ${k}`);
    // The LP often has several optimal plans (same objective). The two front ends can pick different
    // ones, and a 7-day replay then follows a different path, so most (not all) replays must match.
    const lr = model.replay(evenings, c.evening, 7, items, meta, C, e => model.planLP(highs, e, items, meta, K, C).orders.map(o => o[0])).totals;
    gaps.push(Math.abs(lr.total_usd - c.lp_replay.total_usd) / c.lp_replay.total_usd);
    console.log(`${c.date}: objective ${res.objective.toFixed(4)} (Python ${c.lp_objective.toFixed(4)}), ${res.ms.toFixed(0)} ms; ` +
                `7-day replay LP $${lr.total_usd.toFixed(2)} (Python $${c.lp_replay.total_usd.toFixed(2)}), rule $${rr.total_usd.toFixed(2)}`);
  }
  const matching = gaps.filter(g => g < 0.01).length;
  assert(matching * 2 > gaps.length, `only ${matching} of ${gaps.length} LP replays match Python`);
  console.log(`Verified ${evenings.length} evenings x ${items.length} items, file hashes, ${manifest.checks.length} LP solutions against Python (same objective), ` +
              `rule orders and replays (exact) and ${matching} of ${gaps.length} LP replays within 1% (others differ only by tie-breaking).`);
});
