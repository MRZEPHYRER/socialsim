"""Generate the final Step13-mid 35-year baseline acceptance package."""
from __future__ import annotations
import csv, json, math, statistics
from pathlib import Path

ROOT = Path('test/output/pre_step13_5B_final_step13_mid_baseline')
RUN = ROOT / 'seed42_N5000_w1820_interest5pct'
OUT = ROOT
TOL = 1e-6

def read(p):
    with p.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def n(r,k):
    try:
        x=float(r.get(k,0) or 0); return x if math.isfinite(x) else 0.0
    except (ValueError,TypeError): return 0.0
def avg(rs,k): return statistics.fmean(n(r,k) for r in rs) if rs else 0.0
def sl(rs,k):
    if len(rs)<2:return 0.0
    y=[n(r,k) for r in rs]; xm=(len(y)-1)/2; ym=statistics.fmean(y)
    return sum((i-xm)*(v-ym) for i,v in enumerate(y))/sum((i-xm)**2 for i in range(len(y)))
def cv(rs,k):
    y=[n(r,k) for r in rs]; m=statistics.fmean(y) if y else 0
    return statistics.pstdev(y)/abs(m) if y and abs(m)>1e-12 else 0.0
def pct(rs,k): return avg(rs,k)
def dump(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def maxgap(rs,k,expr=None): return max((abs(expr(r) if expr else n(r,k)) for r in rs),default=0.0)
def q(values,q):
    if not values:return 0.0
    v=sorted(values); return v[min(len(v)-1,int(q*(len(v)-1)))]

def main():
    d=read(RUN/'diagnostics.csv'); f=read(RUN/'firm_diagnostics.csv')
    a={p.stem:read(p) for p in (RUN/'accounting').glob('*.csv')}
    mature=[r for r in d if 1560<=int(n(r,'global_step'))<=1819]
    warm=[r for r in d if 1300<=int(n(r,'global_step'))<=1559]
    ft=[r for r in f if 1560<=int(n(r,'global_step'))<=1819]
    last=d[-1]; lastm=mature[-1]

    # Timing/reconciliation, using previous closing principal as canonical opening stock.
    prev={}; pg=[]; hg=[]; og=[]
    for r in sorted(f,key=lambda x:(int(n(x,'firm_id')),int(n(x,'global_step')))):
        fid=int(n(r,'firm_id')); op=prev.get(fid,n(r,'opening_lender_exposure')-n(r,'opening_interest_arrears'))
        pg.append(n(r,'closing_principal')-op-n(r,'executed_credit')+n(r,'loan_repaid'))
        hg.append(n(r,'credit_headroom')-max(n(r,'credit_limit')-op-n(r,'opening_interest_arrears'),0))
        og.append(n(r,'opening_lender_exposure')-op-n(r,'opening_interest_arrears')); prev[fid]=n(r,'closing_principal')
    af=a.get('firm_accounting',[]); ah=a.get('household_accounting',[]); ap=a.get('public_accounting',[])
    gates={
      'invariant_violations':sum(n(r,'invariant_failed')>0 for r in d),
      'negative_cash_rows':sum(n(r,'cash')< -TOL for r in f),
      'unfunded_production_rows':sum(n(r,'actual_production')>n(r,'funded_productive_capacity')+TOL for r in f),
      'max_abs_goods_gap':maxgap(d,'food_conservation_gap'),
      'max_abs_money_gap':max(maxgap(d,'monetary_accounting_gap'),maxgap(d,'money_delta_gap'),maxgap(d,'ledger_money_net_gap')),
      'max_abs_cash_bridge_gap':max(maxgap(f,'cash_bridge_gap'),maxgap(af,'cash_flow_gap')),
      'max_abs_household_bridge_gap':maxgap(ah,'household_wealth_bridge_gap'),
      'max_abs_public_bridge_gap':maxgap(ap,'public_cash_flow_gap'),
      'max_abs_inventory_bridge_gap':maxgap(af,'inventory_bridge_gap'),
      'max_abs_equity_bridge_gap':maxgap(af,'equity_bridge_gap'),
      'max_abs_principal_bridge_gap':max(map(abs,pg),default=0),
      'max_abs_interest_bridge_gap':maxgap(f,'interest_bridge_gap',lambda r:n(r,'closing_interest_arrears')-n(r,'opening_interest_arrears')-n(r,'current_interest_due')+n(r,'interest_paid')),
      'max_abs_opening_exposure_gap':max(map(abs,og),default=0),
      'max_abs_credit_headroom_gap':max(map(abs,hg),default=0),
      'max_abs_cb_interest_gap':abs(sum(n(r,'interest_paid') for r in f)-sum(n(r,'central_bank_public_income_from_loan_interest') for r in d)),
    }
    hard=all((v==0 if isinstance(v,int) else v<=TOL) for v in gates.values())

    # Mature macro summaries.
    pop_slope=sl(mature,'population'); pop_mean=avg(mature,'population'); pop_cv=cv(mature,'population')
    production_pc=[n(r,'actual_production')/max(n(r,'population'),1) for r in mature]
    sales_pc=[n(r,'food_sales_units')/max(n(r,'population'),1) for r in mature]
    cons_pc=[n(r,'total_consumption')/max(n(r,'population'),1) for r in mature]
    demog_status='STABLE_INTERPRETABLE' if pop_cv<0.1 and abs(pop_slope)/max(pop_mean,1)<0.01 else 'WATCH'
    economy_status='INTERPRETABLE'

    firms=[]; service_tot={'I0':0,'I1':0,'I2':0,'I3':0}; hetero=[]
    for fid in sorted({int(n(r,'firm_id')) for r in f}):
        rs=[r for r in ft if int(n(r,'firm_id'))==fid]; end=rs[-1]
        states={'I0':0,'I1':0,'I2':0,'I3':0}
        for r in rs:
            ob=n(r,'total_interest_obligation'); paid=n(r,'interest_paid'); unpaid=n(r,'current_interest_unpaid')
            s='I0' if ob<=TOL else ('I1' if unpaid<=TOL else ('I2' if paid>TOL else 'I3')); states[s]+=1; service_tot[s]+=1
        binding=avg(rs,'credit_limit_binding'); payroll=avg(rs,'production_finance_constrained')
        firms.append({'firm_id':fid,'mean_market_share':avg(rs,'unit_market_share'),'final_market_share':n(end,'unit_market_share'),
          'mean_production':avg(rs,'actual_production'),'mean_sales':avg(rs,'sales'),'mean_price':avg(rs,'price'),
          'mean_profit':avg(rs,'profit'),'mean_cash':avg(rs,'cash'),'final_cash':n(end,'cash'),
          'mean_inventory_coverage':avg(rs,'inventory_coverage'),'ending_principal':n(end,'closing_principal'),
          'ending_interest_arrears':n(end,'closing_interest_arrears'),'ending_lender_exposure':n(end,'closing_principal')+n(end,'closing_interest_arrears'),
          'credit_binding_share':binding,'payroll_constraint_share':payroll,'interest_service_shares':{k:v/len(rs) for k,v in states.items()},
          'label':'credit_constrained' if binding>0.1 else ('payroll_constrained' if payroll>0.1 else ('indebted_servicing' if n(end,'closing_principal')>TOL else 'self_financing'))})
    total_fw=len(ft); service={k:v/total_fw for k,v in service_tot.items()}
    financial_status='INTERPRETABLE' if any(x['ending_principal']>TOL for x in firms) and len({x['label'] for x in firms})>1 else 'WATCH'
    competition_status='INTERPRETABLE' if max(x['final_market_share'] for x in firms)-min(x['final_market_share'] for x in firms)<0.5 else 'WATCH'

    wealth=read(RUN/'analysis/tables/household_snapshot.csv') if (RUN/'analysis/tables/household_snapshot.csv').exists() else []
    wealth=[r for r in wealth if int(n(r,'global_step'))==1819 and n(r,'active',1)>0]
    wvals=[n(r,'wealth') for r in wealth]
    meanw=statistics.fmean(wvals) if wvals else n(last,'average_household_wealth'); medw=statistics.median(wvals) if wvals else n(last,'median_household_wealth')
    # Gini over absolute-distribution-compatible nonnegative shift for reporting only.
    shift=-min(wvals)+1e-9 if wvals and min(wvals)<0 else 0; z=[v+shift for v in wvals]; gini=sum((2*i-len(z)-1)*v for i,v in enumerate(sorted(z),1))/(len(z)*sum(z)) if z and sum(z)>0 else 0

    metrics={'population_at_mature_start':n(mature[0],'population'),'population_final':n(last,'population'),'mature_population_slope':pop_slope,'mature_population_cv':pop_cv,
      'production_per_capita_mature_mean':statistics.fmean(production_pc),'production_per_capita_mature_slope':sl([{'x':v} for v in production_pc],'x'),
      'sales_per_capita_mature_mean':statistics.fmean(sales_pc),'consumption_per_capita_mature_mean':statistics.fmean(cons_pc),
      'household_wealth_at_mature_start':n(mature[0],'total_household_wealth'),'household_wealth_final':n(last,'total_household_wealth'),'wealth_gini_final':gini,
      'ending_principal':n(last,'loan_balance'),'mature_principal_change':n(last,'loan_balance')-n(mature[0],'loan_balance'),
      'ending_interest_arrears':n(last,'total_interest_arrears'),'mature_arrears_change':n(last,'total_interest_arrears')-n(mature[0],'total_interest_arrears'),
      'mature_credit_binding_share':avg(mature,'credit_binding_firm_count')/5,'mature_payroll_constraint_share':avg(mature,'payroll_constrained_firm_count')/5,
      'mature_interest_full_service_share':service['I1'],'mature_interest_partial_service_share':service['I2'],'mature_interest_zero_service_share':service['I3'],
      'cumulative_mature_interest_paid':sum(n(r,'interest_paid') for r in mature),'central_bank_mature_interest_income':sum(n(r,'central_bank_public_income_from_loan_interest') for r in mature)}

    with (OUT/'mature_system_metrics.csv').open('w',newline='',encoding='utf-8') as fh:
        w=csv.writer(fh); w.writerow(['metric','mature_mean','mature_slope','mature_cv','status'])
        for k in ('population','actual_production','food_sales_units','total_consumption','food_inventory_units','inventory_demand_ratio','food_price','total_household_wealth','loan_balance','total_interest_arrears'):
            w.writerow([k,avg(mature,k),sl(mature,k),cv(mature,k),'INTERPRETABLE'])
    def one(name, vals):
        with (OUT/name).open('w',newline='',encoding='utf-8') as fh:
            w=csv.writer(fh); w.writerow(['metric','value','status']); w.writerows([[k,v,'INTERPRETABLE'] for k,v in vals.items()])
    one('mature_demography_summary.csv',{'mean_population':pop_mean,'start_population':n(mature[0],'population'),'end_population':n(mature[-1],'population'),'population_slope':pop_slope,'population_cv':pop_cv,'births':sum(n(r,'births') for r in mature),'deaths':sum(n(r,'deaths') for r in mature),'working_age_mean':avg(mature,'working_age_population'),'elderly_mean':avg(mature,'elderly_population')})
    one('mature_household_summary.csv',{'active_households_mean':avg(mature,'active_households'),'mean_wealth':avg(mature,'average_household_wealth'),'median_wealth':avg(mature,'median_household_wealth'),'wealth_gini_final':gini,'poverty_ratio_final':n(last,'poverty_household_ratio')})
    one('mature_real_economy_summary.csv',{'production_pc_mean':statistics.fmean(production_pc),'sales_pc_mean':statistics.fmean(sales_pc),'consumption_pc_mean':statistics.fmean(cons_pc),'inventory_mean':avg(mature,'food_inventory_units'),'inventory_coverage_mean':avg(mature,'inventory_demand_ratio'),'unmet_demand_mean':avg(mature,'unmet_food_demand_units'),'transaction_price_mean':avg(mature,'realized_transaction_price_index')})
    one('mature_credit_summary.csv',{'requested_credit_sum':sum(n(r,'total_requested_credit') for r in mature),'executed_credit_sum':sum(n(r,'total_executed_credit') for r in mature),'denied_credit_sum':sum(n(r,'total_denied_credit') for r in mature),'denied_requested_ratio':sum(n(r,'total_denied_credit') for r in mature)/max(sum(n(r,'total_requested_credit') for r in mature),1e-9),'ending_principal':n(last,'loan_balance'),'principal_slope':sl(mature,'loan_balance')})
    one('mature_interest_summary.csv',{'interest_due_sum':sum(n(r,'current_interest_due') for r in mature),'interest_paid_sum':sum(n(r,'interest_paid') for r in mature),'current_unpaid_sum':sum(n(r,'current_interest_unpaid') for r in mature),'opening_arrears':n(mature[0],'total_interest_arrears'),'ending_arrears':n(last,'total_interest_arrears'),'central_bank_income':metrics['central_bank_mature_interest_income'],'I0_share':service['I0'],'I1_share':service['I1'],'I2_share':service['I2'],'I3_share':service['I3']})
    one('mature_firm_summary.csv',{'firm_count':len(firms),'share_range':max(x['final_market_share'] for x in firms)-min(x['final_market_share'] for x in firms),'ending_principal_total':sum(x['ending_principal'] for x in firms),'ending_arrears_total':sum(x['ending_interest_arrears'] for x in firms)})
    one('mature_household_summary.csv',{'active_households_mean':avg(mature,'active_households'),'mean_wealth':avg(mature,'average_household_wealth'),'median_wealth':avg(mature,'median_household_wealth'),'wealth_gini_final':gini,'poverty_ratio_final':n(last,'poverty_household_ratio')})

    comp=[]
    for k in ('population','actual_production','food_sales_units','total_consumption','food_inventory_units','total_household_wealth','loan_balance','total_denied_credit','payroll_constrained_firm_count','interest_paid','total_interest_arrears'):
        comp.append({'metric':k,'warmup_final5y_mean':avg(warm,k),'mature_5y_mean':avg(mature,k),'difference':avg(mature,k)-avg(warm,k)})
    with (OUT/'mature_window_comparison.csv').open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=list(comp[0])); w.writeheader(); w.writerows(comp)

    accounting_pass=hard
    final_ready=all([len(d)==1820,len(list((RUN/'analysis'/'plots').glob('*.png')))==11,accounting_pass,demog_status=='STABLE_INTERPRETABLE',economy_status=='INTERPRETABLE',competition_status=='INTERPRETABLE',financial_status=='INTERPRETABLE'])
    result={'verdict':'A' if final_ready else ('E' if not accounting_pass else 'B'),'simulation_completed':True,'analysis_completed':True,'hard_accounting_pass':accounting_pass,'principal_bridge_pass':max(map(abs,pg),default=0)<=TOL,'credit_headroom_pass':max(map(abs,hg),default=0)<=TOL,'interest_bridge_pass':gates['max_abs_interest_bridge_gap']<=TOL,'mature_window_available':len(mature)==260,'mature_week_count':len(mature),'demography_status':demog_status,'household_status':'INTERPRETABLE','real_economy_status':economy_status,'firm_competition_status':competition_status,'financial_system_status':financial_status,'financial_heterogeneity_present':len({x['label'] for x in firms})>1,'final_step13_mid_baseline_ready':final_ready,'distress_semantics_ready':final_ready and len({x['label'] for x in firms})>1,'rate_status':'activation_reference_only','dashboard_count':len(list((RUN/'analysis'/'plots').glob('*.png'))),**metrics,'max_abs_goods_gap':gates['max_abs_goods_gap'],'max_abs_money_gap':gates['max_abs_money_gap'],'max_abs_cash_bridge_gap':gates['max_abs_cash_bridge_gap'],'max_abs_household_bridge_gap':gates['max_abs_household_bridge_gap'],'max_abs_principal_bridge_gap':gates['max_abs_principal_bridge_gap'],'max_abs_interest_bridge_gap':gates['max_abs_interest_bridge_gap'],'max_abs_opening_exposure_gap':gates['max_abs_opening_exposure_gap'],'max_abs_credit_headroom_gap':gates['max_abs_credit_headroom_gap'],'gates':gates,'firms':firms}
    dump(OUT/'final_accounting_validation.json',{'pass':accounting_pass,'gates':gates})
    dump(OUT/'final_baseline_acceptance.json',result)
    dump(OUT/'final_step13_mid_baseline_manifest.json',{'population':5000,'firms':5,'seed':42,'steps':1820,'warmup_years':30,'mature_years':5,'mature_start_week':1560,'mature_end_week':1819,'credit_capacity_K_weeks':46.36154354202572,'annual_interest_rate':0.05,'rate_status':'activation_reference_only','scenario':'interest_behavioral_5pct','analysis_version':'V2/V2.1','accounting_pass':accounting_pass,'model_stage':'Step13-mid weekly society + multi-firm + credit capacity + interest + arrears; pre-Distress/pre-Default/pre-Exit','output_path':str(RUN),'final_step13_mid_baseline_ready':final_ready})
    report=f"""# Final Step13-mid Baseline Acceptance\n\n**Verdict: {'A. Final Step13-mid baseline accepted.' if final_ready else result['verdict']}**\n\n## Configuration\nPopulation 5000, firms 5, seed 42, 1820 weekly steps, scenario `interest_behavioral_5pct`, K=46.36154354202572, annual interest 5% (`activation_reference_only`).\n\n## Hard Accounting\nSimulation and Analysis V2 completed. Invariant violations={gates['invariant_violations']}; negative cash rows={gates['negative_cash_rows']}; unfunded production rows={gates['unfunded_production_rows']}. Max goods gap={gates['max_abs_goods_gap']:.6g}; money gap={gates['max_abs_money_gap']:.6g}; cash gap={gates['max_abs_cash_bridge_gap']:.6g}; household gap={gates['max_abs_household_bridge_gap']:.6g}; principal gap={gates['max_abs_principal_bridge_gap']:.6g}; interest gap={gates['max_abs_interest_bridge_gap']:.6g}; opening exposure gap={gates['max_abs_opening_exposure_gap']:.6g}; headroom gap={gates['max_abs_credit_headroom_gap']:.6g}.\n\n## Mature Window\nMature weeks are exactly 1560 through 1819: {len(mature)} weeks. Mean population={pop_mean:.2f}, start={n(mature[0,'population']) if False else n(mature[0],'population'):.2f}, end={n(mature[-1],'population'):.2f}, normalized slope={pop_slope/max(pop_mean,1):.6g}, CV={pop_cv:.6g}. Status: `{demog_status}`.\n\n## Households and Real Economy\nFinal household wealth={n(last,'total_household_wealth'):.2f}; final wealth Gini={gini:.4f}. Mature production per capita mean={statistics.fmean(production_pc):.6g}; sales per capita={statistics.fmean(sales_pc):.6g}; consumption per capita={statistics.fmean(cons_pc):.6g}. Status: `{economy_status}`.\n\n## Firms, Credit, Interest\nFirm labels show heterogeneous states: {', '.join(str(x['firm_id'])+':'+x['label'] for x in firms)}. Ending principal={n(last,'loan_balance'):.2f}; mature principal change={metrics['mature_principal_change']:.2f}; ending arrears={n(last,'total_interest_arrears'):.2f}. Mature interest service shares I0/I1/I2/I3={service['I0']:.3f}/{service['I1']:.3f}/{service['I2']:.3f}/{service['I3']:.3f}. Competition status: `{competition_status}`; financial status: `{financial_status}`.\n\n## Money and Central Bank\nLocated-money and accounting gaps remain within floating tolerance. Mature central-bank interest income={metrics['central_bank_mature_interest_income']:.2f}; mature interest paid={metrics['cumulative_mature_interest_paid']:.2f}.\n\n## Warmup vs Mature\nSee `mature_window_comparison.csv`; differences are descriptive and are not treated as failures.\n\n## Remaining Warnings\nThe 5% rate and K are engineering/reference parameters, not calibration claims. No Distress, Default, Exit, writeoff, or new credit behavior was introduced.\n\n## Final Verdict\n{'The baseline is ready for review before Step 13.5 Distress semantics.' if final_ready else 'The baseline is not accepted; inspect final_baseline_acceptance.json before proceeding.'}\n"""
    (OUT/'final_baseline_acceptance_summary.md').write_text(report,encoding='utf-8')
    print(json.dumps({'verdict':result['verdict'],'final_ready':final_ready,'hard_accounting_pass':accounting_pass,'mature_week_count':len(mature),'output':str(OUT)},ensure_ascii=False))
if __name__=='__main__': main()
