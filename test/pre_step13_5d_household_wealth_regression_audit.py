"""Focused, non-behavioral audit of the final household wealth distribution."""
from __future__ import annotations
import csv, json, math, statistics
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT=Path('test/output/pre_step13_5D_household_wealth_regression_audit')
RUN=Path('test/output/pre_step13_5B_final_step13_mid_baseline/seed42_N5000_w1820_interest5pct')
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def n(r,k):
    try:return float(r.get(k,0) or 0)
    except:return 0.0
def pct(v,q): return float(np.percentile(np.asarray(v,float),q)) if v else 0.0
def mean(v): return statistics.fmean(v) if v else 0.0
def med(v): return statistics.median(v) if v else 0.0
def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        fields=[]
        for row in rows:
            for key in row:
                if key not in fields: fields.append(key)
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def gini(v):
    v=sorted(x for x in v if x>=0); return sum((2*i-len(v)-1)*x for i,x in enumerate(v,1))/(len(v)*sum(v)) if v and sum(v) else 0

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    h=read(RUN/'analysis'/'tables'/'household_snapshot.csv'); lifecycle=read(RUN/'accounting'/'household_lifecycle.csv')
    wealth=[n(r,'wealth') for r in h]; low=[r for r in h if abs(n(r,'wealth'))<100]; lower=[r for r in h if 100<=abs(n(r,'wealth'))<1000]; established=[r for r in h if n(r,'wealth')>=1000]
    exact={'active_households':len(h),'minimum':min(wealth),'p01':pct(wealth,1),'p05':pct(wealth,5),'p10':pct(wealth,10),'p25':pct(wealth,25),'median':pct(wealth,50),'p75':pct(wealth,75),'p90':pct(wealth,90),'p95':pct(wealth,95),'p99':pct(wealth,99),'maximum':max(wealth),'exact_zero_count':sum(x==0 for x in wealth),'abs_wealth_lt_1_count':sum(abs(x)<1 for x in wealth),'abs_wealth_lt_10_count':sum(abs(x)<10 for x in wealth),'abs_wealth_lt_50_count':sum(abs(x)<50 for x in wealth),'abs_wealth_lt_100_count':sum(abs(x)<100 for x in wealth)}
    bands=[('-inf,-1000',lambda x:x<-1000),('-1000,-100',lambda x:-1000<=x<-100),('-100,-50',lambda x:-100<=x<-50),('-50,-10',lambda x:-50<=x<-10),('-10,0',lambda x:-10<=x<0),('0,10',lambda x:0<=x<10),('10,50',lambda x:10<=x<50),('50,100',lambda x:50<=x<100),('100,500',lambda x:100<=x<500),('500,1000',lambda x:500<=x<1000),('1000+',lambda x:x>=1000)]
    dist=[{'band':name,'count':sum(fn(x) for x in wealth),'share':sum(fn(x) for x in wealth)/len(wealth)} for name,fn in bands]
    exact['low_wealth_share']=len(low)/len(h); exact['negative_wealth_count']=sum(x<0 for x in wealth)
    write_csv(OUT/'household_wealth_exact_distribution.csv',[exact]+dist)
    # Fine bins and repeated-value checks.
    fine=[]
    for width in (5,10,20):
        edges=np.arange(-150,150+width,width); counts,_=np.histogram(wealth,bins=edges)
        for i,c in enumerate(counts):fine.append({'bin_width':width,'lower':edges[i],'upper':edges[i+1],'count':int(c),'share':int(c)/len(wealth)})
    write_csv(OUT/'low_wealth_fine_histogram.csv',fine)
    rounded=[]
    for unit in (0.01,0.1,1,5,10):
        c=Counter(round(x/unit)*unit for x in wealth)
        for value,count in c.most_common(20):rounded.append({'rounding_unit':unit,'value':value,'count':count})
    write_csv(OUT/'household_low_wealth_cohort.csv',[{'cohort':'LOW_WEALTH','count':len(low),'share':len(low)/len(h),'mean_wealth':mean([n(r,'wealth') for r in low]),'median_wealth':med([n(r,'wealth') for r in low]),'mean_income':mean([n(r,'income') for r in low]),'median_income':med([n(r,'income') for r in low]),'mean_consumption':mean([n(r,'consumption') for r in low]),'median_consumption':med([n(r,'consumption') for r in low]),'mean_saving':mean([n(r,'saving') for r in low]),'median_saving':med([n(r,'saving') for r in low]),'mean_workers':mean([n(r,'working_members') for r in low]),'mean_children':mean([n(r,'children') for r in low])},{'cohort':'LOWER_MIDDLE','count':len(lower),'share':len(lower)/len(h),'mean_wealth':mean([n(r,'wealth') for r in lower]),'median_wealth':med([n(r,'wealth') for r in lower]),'mean_income':mean([n(r,'income') for r in lower]),'median_income':med([n(r,'income') for r in lower]),'mean_consumption':mean([n(r,'consumption') for r in lower]),'median_consumption':med([n(r,'consumption') for r in lower]),'mean_saving':mean([n(r,'saving') for r in lower]),'median_saving':med([n(r,'saving') for r in lower]),'mean_workers':mean([n(r,'working_members') for r in lower]),'mean_children':mean([n(r,'children') for r in lower])},{'cohort':'POSITIVE_ESTABLISHED','count':len(established),'share':len(established)/len(h),'mean_wealth':mean([n(r,'wealth') for r in established]),'median_wealth':med([n(r,'wealth') for r in established]),'mean_income':mean([n(r,'income') for r in established]),'median_income':med([n(r,'income') for r in established]),'mean_consumption':mean([n(r,'consumption') for r in established]),'median_consumption':med([n(r,'consumption') for r in established]),'mean_saving':mean([n(r,'saving') for r in established]),'median_saving':med([n(r,'saving') for r in established]),'mean_workers':mean([n(r,'working_members') for r in established]),'mean_children':mean([n(r,'children') for r in established])}])
    write_csv(OUT/'household_wealth_distribution_audit.csv',[{'measure':k,'value':v} for k,v in exact.items()]+rounded)
    # Type composition and within-type rates.
    types=sorted({r.get('household_type','unknown') for r in h}); rows=[]
    for typ in types:
        all_t=[r for r in h if r.get('household_type')==typ]; low_t=[r for r in all_t if abs(n(r,'wealth'))<100]
        rows.append({'household_type':typ,'all_count':len(all_t),'low_count':len(low_t),'low_share_within_type':len(low_t)/len(all_t) if all_t else 0,'share_of_low_cohort':len(low_t)/len(low) if low else 0,'mean_workers':mean([n(r,'working_members') for r in low_t]),'mean_children':mean([n(r,'children') for r in low_t])})
    write_csv(OUT/'household_low_wealth_group_comparison.csv',rows)
    # Plots.
    fig,ax=plt.subplots(figsize=(9,5)); ax.hist(wealth,bins=np.arange(-150,151,5),color='#607d8b'); ax.axvline(-100,color='red',ls='--');ax.axvline(100,color='red',ls='--');ax.set_xlim(-150,150);ax.set_title('Final household wealth: low-wealth zoom');ax.set_xlabel('wealth');ax.set_ylabel('households');fig.tight_layout();fig.savefig(OUT/'wealth_distribution_zoom.png',dpi=140);plt.close(fig)
    flow=np.array([n(r,'income')-n(r,'consumption') for r in h]); fig,ax=plt.subplots(1,2,figsize=(12,4.5)); colors=['#d95f02' if abs(n(r,'wealth'))<100 else '#9e9e9e' for r in h];ax[0].scatter(wealth,flow,c=colors,s=10);ax[0].axvline(-100,ls='--');ax[0].axvline(100,ls='--');ax[0].set_xlabel('wealth');ax[0].set_ylabel('income - consumption');ax[1].scatter(wealth,[n(r,'saving') for r in h],c=colors,s=10);ax[1].set_xlabel('wealth');ax[1].set_ylabel('saving');fig.tight_layout();fig.savefig(OUT/'wealth_netflow_relationship.png',dpi=140);plt.close(fig)
    # Lifecycle events are available, but household wealth history is not.
    ev=[]
    for typ,rs in sorted(__import__('itertools').groupby(sorted(lifecycle,key=lambda r:r.get('event_type','')),key=lambda r:r.get('event_type',''))):
        rs=list(rs); ev.append({'event_type':typ,'count':len(rs),'mean_lifecycle_cash_gap':mean([n(r,'lifecycle_cash_gap') for r in rs]),'max_abs_gap':max([abs(n(r,'lifecycle_cash_gap')) for r in rs],default=0),'mean_wealth_before':mean([n(r,'wealth_before') for r in rs]),'mean_wealth_after':mean([n(r,'wealth_after') for r in rs]),'low_wealth_events':sum(abs(n(r,'wealth_after'))<100 for r in rs)})
    write_csv(OUT/'lifecycle_event_wealth_check.csv',ev)
    # Static source map of direct mutation sites found by source audit.
    source=[('household.py','Household.__init__','wealth = 0.0','initialization','inflow/outflow: none','none','initial state',True),('world.py','leave_household','household.wealth -> pending_household_formation_wealth','separation / adulthood','outflow','world.pending_household_formation_wealth','ledger transfer',True),('world.py','record_household_lifecycle_event','wealth_after - wealth_before ...','lifecycle audit','diagnostic','none','lifecycle diagnostic',True),('world.py','public support settlement','household.wealth += per_household','public support','inflow','CentralBank/PublicSector','ledger transfer',True),('world.py','consumption settlement','household.wealth -= delta','consumption','outflow','Firm/Public seller','ledger transfer',True),('world.py','consumption refund','household.wealth += refund','consumption refund','inflow','Firm/Public seller','ledger transfer',True),('world.py','wage/dividend/inheritance paths','household.wealth += ...','income/transfer/inheritance','inflow','Firm/household/public','ledger transfer',True)]
    write_csv(OUT/'household_wealth_source_map.csv',[{'file':a,'function':b,'line_code_location':c,'mutation_type':d,'inflow_outflow':e,'expected_counterparty':f,'ledger_accounting_representation':g,'lifecycle_relevance':h} for a,b,c,d,e,f,g,h in source])
    (OUT/'household_negative_wealth_semantics.md').write_text("""# Negative Wealth Semantics\n\n`Household.wealth` is the household cash/wealth balance used in settlement, and the current code permits it to become negative; the final snapshot contains no exact-zero clipping mass and includes negative values. It is therefore not a strictly nonnegative spendable-cash constraint.\n\nAffordability is an intermediate budget calculation. A negative affordable-consumption value can coexist with nonnegative actual consumption when minimum/necessary consumption and public-support/settlement paths are applied separately; the current outputs do not contain enough household-level fields to prove the exact per-household sequence. This is an evidence gap, not a claim that the semantics are correct.\n""",encoding='utf-8')
    (OUT/'household_lifecycle_wealth_audit.md').write_text(f"""# Household Lifecycle Wealth Audit\n\nLifecycle accounting CSV is available with {len(lifecycle)} events and event-level `lifecycle_cash_gap`. Aggregate event rows can be checked, but no before/after household wealth history keyed to every active household exists. New-household starting wealth and formation age for final low-wealth households cannot be reconstructed from this output.\n\nThe existing lifecycle event records should be used for targeted event-window instrumentation; this audit does not infer that lifecycle events caused the low-wealth band.\n""",encoding='utf-8')
    (OUT/'household_micro_bridge_evidence_gap.json').write_text(json.dumps({'micro_household_bridge_available':False,'available_components':['final wealth','income','consumption','saving','household lifecycle event gaps'],'missing_components':['opening/closing wealth for every household-week','wages by household','dividends by household','public support by household','inheritance by household','transfers in/out by household'],'required_minimum_instrumentation':['household weekly opening wealth','household weekly wage/dividend/support/transfer inflows','household weekly consumption','household weekly closing wealth','lifecycle event linkage'],'do_not_claim_pass':True},indent=2),encoding='utf-8')
    (OUT/'historical_regression_comparison.md').write_text('# Historical Regression Comparison\n\nNo compatible older smooth-distribution run or weekly household wealth history was found in the current output package. Git/source history was not used to infer a regression. A targeted N=500 reproduction or a future lightweight household-history run is recommended only after review.\n',encoding='utf-8')
    result={'verdict':'G. INSUFFICIENT_EVIDENCE','active_households':len(h),'exact_zero_count':exact['exact_zero_count'],'abs_wealth_lt_1_count':exact['abs_wealth_lt_1_count'],'abs_wealth_lt_10_count':exact['abs_wealth_lt_10_count'],'abs_wealth_lt_50_count':exact['abs_wealth_lt_50_count'],'abs_wealth_lt_100_count':exact['abs_wealth_lt_100_count'],'low_wealth_share':len(low)/len(h),'modal_low_wealth_bin':float(max(fine,key=lambda r:r['count'])['lower']),'modal_low_wealth_bin_count':int(max(r['count'] for r in fine)),'low_wealth_mean_income':mean([n(r,'income') for r in low]),'low_wealth_median_income':med([n(r,'income') for r in low]),'low_wealth_mean_consumption':mean([n(r,'consumption') for r in low]),'low_wealth_median_consumption':med([n(r,'consumption') for r in low]),'low_wealth_mean_saving':mean([n(r,'saving') for r in low]),'low_wealth_median_saving':med([n(r,'saving') for r in low]),'low_wealth_budget_constrained_share':None,'low_wealth_public_support_share':None,'dominant_low_wealth_household_type':Counter(r.get('household_type') for r in low).most_common(1)[0][0] if low else None,'dominant_low_wealth_worker_count':int(Counter(int(n(r,'working_members')) for r in low).most_common(1)[0][0]) if low else None,'negative_wealth_count':sum(x<0 for x in wealth),'new_household_initialization_semantics':'not_reconstructable_from_final_snapshot','micro_household_bridge_available':False,'historical_low_wealth_series_available':False,'household_wealth_structure_ready_for_distress':False,'economic_behavior_changed':False,'full_long_rerun_required':False}
    (OUT/'acceptance_summary.md').write_text(f"""# PRE-STEP 13.5D Household Wealth Regression Audit\n\n**Verdict: G. INSUFFICIENT_EVIDENCE**\n\nThe exact distribution confirms a low-wealth band but not an exact zero point mass: active households={len(h)}, exact zero={exact['exact_zero_count']}, `|wealth|<100`={len(low)} ({len(low)/len(h):.2%}). Low-wealth mean income={result['low_wealth_mean_income']:.3f}, median={result['low_wealth_median_income']:.3f}; mean consumption={result['low_wealth_mean_consumption']:.3f}; mean saving={result['low_wealth_mean_saving']:.3f}.\n\nThe available snapshot lacks household-level necessary/desired/affordable consumption, public support, security ratio, economic pressure, wage/dividend allocation and historical wealth. Therefore it cannot distinguish an economic low-wealth regime from a micro-accounting or lifecycle initialization regression. The micro household bridge and historical low-wealth persistence series are unavailable.\n\nNo household behavior, parameters, lifecycle rules or long simulation were changed. Distress remains paused.\n""",encoding='utf-8')
    (OUT/'audit_result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
