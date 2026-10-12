import {report as panelReport} from '../showdown-panel/report.mjs';
import {confirm} from './statistics.mjs';

export function report(plan, records, totals) {
  const summary = panelReport(plan, records, totals);
  return {...summary, schema: 'showdown-confirmation-report-1', confirmation: confirm(summary.matrix, plan.confirmation),
    interpretation: 'Fixed fresh development confirmation of unchanged tactical-v1; nominal approximate paired bounds, no final, ladder, learning or G2/G3 claim.'};
}
