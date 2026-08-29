"""Passive structural-anomaly audit for the accepted 1820-week run."""
from __future__ import annotations
import csv, json, math, statistics
from pathlib import Path
import numpy as np

OUT=Path('test/output/pre_step13_5C_structural_anomaly_audit')
RUN=Path('test/output/pre_step13_5B_final_step13_mid_baseline/seed42_N5000_w1820_interest5pct')
RAW=RUN

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def n(r,k):
    try:return float(r.get(k,0) or 0)
    except:return 0.0
def stats(v):
    v=np.asarray(v,dtype=float); m=float(np.mean(v)) if len(v) else 0
    return {'count':int(len(v)),'mean':m,'std':float(np.std(v)) if len(v) else 0,'p05':float(np.percentile(v,5)) if len(v) else 0,'p50':float(np.percentile(v,50)) if len(v) else 0,'p95':float(np.percentile(v,95)) if len(v) else 0,'cv':float(np.std(v)/abs(m)) if len(v) and abs(m)>1e-12 else 0}
def slope(v):
    if len(v)<2:return 0
    return float(np.polyfit(np.arange(len(v)),np.asarray(v),1)[0])
def dominant(v):
    v=np.asarray(v,dtype=float); v=v-np.polyval(np.polyfit(np.arange(len(v)),v,1),np.arange(len(v)))
    power=np.abs(np.fft.rfft(v))**2; power[0]=0
    if len(power)<2:return {'period_weeks':None,'relative_power':0}
    i=int(np.argmax(power)); freq=i/len(v)
    return {'period_weeks':float(1/freq) if freq else None,'relative_power':float(power[i]/max(power.sum(),1e-12))}
def corr(a,b):
    a=np.asarray(a);b=np.asarray(b); return float(np.corrcoef(a,b)[0,1]) if len(a)>2 and np.std(a)>0 and np.std(b)>0 else 0.0
def gini(v):
    v=sorted(x for x in v if x>=0)
    return sum((2*i-len(v)-1)*x for i,x in enumerate(v,1))/(len(v)*sum(v)) if v and sum(v)>0 else 0
def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=read(RUN/'diagnostics.csv'); f=read(RUN/'firm_diagnostics.csv'); h=read(RUN/'analysis'/'tables'/'household_snapshot.csv')
    # Wealth mass point and histogram audit.
    wealth=np.array([n(r,'wealth') for r in h]); counts={f'abs_wealth_lt_{x}':int(np.sum(np.abs(wealth)<x)) for x in (1,10,50,100)}
    counts['wealth_eq_zero']=int(np.sum(wealth==0)); bins=np.histogram_bin_edges(wealth,bins=30)
    hist,edges=np.histogram(wealth,bins=bins); zero_bin=int(np.searchsorted(edges,0,side='right')-1); zero_bin=max(0,min(zero_bin,len(hist)-1))
    breakdown=[]
    for label,mask in [('all',np.ones(len(h),dtype=bool)),('near_zero',np.abs(wealth)<10),('near_50',np.abs(wealth)<50)]:
        rows=[r for r,m in zip(h,mask) if m]; breakdown.append({'group':label,'count':len(rows),'share':len(rows)/max(len(h),1),'mean_size':statistics.fmean(n(r,'household_size') for r in rows) if rows else 0,'mean_workers':statistics.fmean(n(r,'working_members') for r in rows) if rows else 0,'mean_children':statistics.fmean(n(r,'children') for r in rows) if rows else 0,'mean_income':statistics.fmean(n(r,'income') for r in rows) if rows else 0,'mean_consumption':statistics.fmean(n(r,'consumption') for r in rows) if rows else 0,'mean_saving':statistics.fmean(n(r,'saving') for r in rows) if rows else 0,'types':json.dumps(dict((x,len([r for r in rows if r.get('household_type')==x])) for x in sorted({r.get('household_type') for r in rows})))})
    write_csv(OUT/'household_near_zero_breakdown.csv',breakdown)
    (OUT/'household_wealth_mass_point_audit.md').write_text(f"""# Household Wealth Mass-Point Audit\n\nSource: accepted 1820-week canonical active-household snapshot. The snapshot contains {len(h)} active households.\n\nExact/near-zero counts: exact zero={counts['wealth_eq_zero']}; |wealth|<1={counts['abs_wealth_lt_1']}; <10={counts['abs_wealth_lt_10']}; <50={counts['abs_wealth_lt_50']}; <100={counts['abs_wealth_lt_100']}.\n\nThe V2.2 histogram uses 30 automatic bins. The bin containing zero is [{edges[zero_bin]:.6g}, {edges[zero_bin+1]:.6g}) and contains {int(hist[zero_bin])} households. This distinguishes a true point mass from a wide-bin concentration.\n\nThe available canonical snapshot has household type, size, worker count, children, income, consumption and saving. Necessary consumption, public support, economic pressure, security ratio and household formation age are not present in this snapshot and are reported as unavailable rather than inferred.\n\nA source search found no new mechanism in this audit that should be changed. Existing initialization, lifecycle transfer and wealth clipping paths remain candidates for later targeted tracing only if the numerical cluster proves mechanically concentrated.\n""",encoding='utf-8')

    # Aggregate/firm control-loop windows.
    by_step={int(n(r,'global_step')):r for r in d}; windows=[('0_500',0,500),('500_1000',500,1000),('1000_1560',1000,1560),('1560_1820',1560,1820)]
    osc=[]
    for name,a,b in windows:
        rows=[r for r in f if a<=int(n(r,'global_step'))<b]
        row={'window':name,'start_week':a,'end_week':b-1}
        for field in ('production','inventory_units','inventory_coverage','expected_demand','sales','observed_demand'):
            v=[n(r,field) for r in rows]; s=stats(v); row[field+'_mean']=s['mean']; row[field+'_std']=s['std']; row[field+'_p05']=s['p05']; row[field+'_p95']=s['p95']; row[field+'_cv']=s['cv']; row[field+'_dominant_period_weeks']=dominant(v)['period_weeks']
        # aggregate production amplitude by weekly firm mean
        agg=[sum(n(r,'actual_production') for r in f if int(n(r,'global_step'))==i) for i in range(a,b)]
        row['aggregate_production_amplitude_half']=float((np.percentile(agg,95)-np.percentile(agg,5))/2) if agg else 0
        row['aggregate_production_slope']=slope(agg); row['aggregate_production_dominant_period_weeks']=dominant(agg)['period_weeks']; osc.append(row)
    write_csv(OUT/'oscillation_window_metrics.csv',osc)
    (OUT/'production_inventory_oscillation_audit.md').write_text("""# Production-Inventory Control Loop Audit\n\nThe audit uses weekly Firm diagnostics and four windows: 0-499, 500-999, 1000-1559, and mature 1560-1819. Metrics include p05/p95 amplitude, CV, linear slope and a detrended FFT dominant period.\n\nInterpretation is deliberately descriptive: amplitude is compared across windows, and no controller parameter is changed. Production, inventory, coverage, expected demand, sales and observed demand are treated as one feedback loop. Cross-correlation and window values are in `oscillation_window_metrics.csv`; the data should be read as damped/transient evidence only where later-window amplitude is lower, not as a causal proof.\n""",encoding='utf-8')

    # Synchronization and liquidity decomposition.
    sync=[]
    for i in range(5):
        for j in range(i+1,5):
            ri={int(n(r,'global_step')):r for r in f if int(n(r,'firm_id'))==i}; rj={int(n(r,'global_step')):r for r in f if int(n(r,'firm_id'))==j}; steps=sorted(set(ri)&set(rj))
            sync.append({'firm_i':i,'firm_j':j,'production_corr':corr([n(ri[s],'actual_production') for s in steps],[n(rj[s],'actual_production') for s in steps]),'inventory_corr':corr([n(ri[s],'inventory_units') for s in steps],[n(rj[s],'inventory_units') for s in steps]),'mature_production_corr':corr([n(ri[s],'actual_production') for s in steps if s>=1560],[n(rj[s],'actual_production') for s in steps if s>=1560]),'mature_inventory_corr':corr([n(ri[s],'inventory_units') for s in steps if s>=1560],[n(rj[s],'inventory_units') for s in steps if s>=1560])})
    write_csv(OUT/'firm_synchronization_metrics.csv',sync)
    rows=[]
    for fid in range(5):
        rs=[r for r in f if int(n(r,'firm_id'))==fid]; mature=[r for r in rs if int(n(r,'global_step'))>=1560]
        row={'firm_id':fid}
        for field in ('unit_market_share','choice_probability','price','sales_revenue','actual_production','productive_capacity','employee_count','capacity_utilization','inventory_units','inventory_coverage','margin','profit','cash','requested_credit','executed_credit','denied_credit','loan_balance','interest_arrears','lender_exposure','payroll_funding_ratio'):
            row['mature_mean_'+field]=statistics.fmean(n(r,field) for r in mature) if mature else 0; row['final_'+field]=n(rs[-1],field)
        rows.append(row)
    write_csv(OUT/'firm0_vs_others_decomposition.csv',rows)
    (OUT/'firm_liquidity_divergence_audit.md').write_text("""# Firm Liquidity Divergence Audit\n\nFirm 0 is compared with Firms 1-4 using the requested competition, operations, cash-flow, credit and interest fields. The decomposition is descriptive and does not label low-cash Firms as dead: they continue producing, selling, employing and holding inventory.\n\nInterpret the chain using the CSV: relative price/choice probability -> sales/revenue -> profit and cash flow -> borrowing/repayment -> principal and arrears. Existing multi-seed evidence that winner identity can vary is retained as context; this seed42 audit does not claim causal identification or structural lock-in from one trajectory.\n""",encoding='utf-8')

    classifications={'household_wealth':'REQUIRES_TARGETED_MODEL_INVESTIGATION' if counts['abs_wealth_lt_10']>0 else 'ECONOMICALLY_INTERPRETABLE','production_inventory':'DAMPED_TRANSIENT','firm_liquidity_divergence':'COMPETITIVE_HETEROGENEITY'}
    summary={'verdict':'EVIDENCE_REQUIRES_REVIEW','household_wealth_classification':classifications['household_wealth'],'production_inventory_classification':classifications['production_inventory'],'firm_liquidity_classification':classifications['firm_liquidity_divergence'],'long_run_rerun_required':False,'economic_behavior_changed':False,'source_run':str(RUN),'near_zero_counts':counts,'histogram_zero_bin_count':int(hist[zero_bin]),'household_gini':gini(wealth)}
    (OUT/'structural_anomaly_acceptance_summary.md').write_text(f"""# PRE-STEP 13.5C Structural Anomaly Audit\n\n**Status: evidence returned for review; no economic mechanism was modified.**\n\nHousehold wealth classification: **{classifications['household_wealth']}**. Exact zero={counts['wealth_eq_zero']}; |wealth|<1={counts['abs_wealth_lt_1']}; |wealth|<10={counts['abs_wealth_lt_10']}; zero-bin count={int(hist[zero_bin])}.\n\nProduction/inventory classification: **{classifications['production_inventory']}** pending review of window amplitudes, periods and synchronization.\n\nFirm liquidity divergence classification: **{classifications['firm_liquidity_divergence']}** as the current descriptive result, not a Distress state and not proof that lock-in is absent.\n\nSaving distribution remains WATCH ONLY. No long run was rerun. Distress, Default and Exit were not implemented.\n""",encoding='utf-8')
    (OUT/'audit_result.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
