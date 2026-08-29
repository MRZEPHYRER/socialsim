"""Step17.I - short member-level low-income mechanism attribution audit."""
from __future__ import annotations
import hashlib, importlib.util, json, math, shutil
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'test/output/step17_i_low_income_mechanism_attribution'
HOUT=ROOT/'test/output/step17_h_structural_low_tail_decomposition'
CHECKPOINT=ROOT/'test/output/mature_population_checkpoint/mature_population_week_2600.pkl'
EPS=1e-8; LOW=.25; NONLOW=1.0; WINDOW=26; SAMPLE_LIMIT=32; EVENT_LIMIT=64

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def num(value, default=np.nan):
    try:
        x=float(value); return x if math.isfinite(x) else default
    except (TypeError,ValueError): return default

def mean_or_nan(values):
    x=[num(v) for v in values]; x=[v for v in x if math.isfinite(v)]; return float(np.mean(x)) if x else np.nan

def ratio(a,b):
    a,b=num(a),num(b); return a/b if math.isfinite(a) and math.isfinite(b) and abs(b)>EPS else np.nan

def members_for(world,household):
    out=[]; seen=set()
    for pid in list(getattr(household,'parents',[]))+list(getattr(household,'children',[])):
        if pid in seen: continue
        seen.add(pid); p=world.get_person_by_id(pid)
        if p is not None and getattr(p,'alive',False): out.append(p)
    return out

def household_type(household,members):
    pa=len(getattr(household,'parents',[])); ch=len(members)-pa
    if pa==2: return {0:'couple_no_children',1:'couple_one_child',2:'couple_two_children'}.get(ch,'couple_three_plus_children')
    if pa==1: return {0:'single_adult',1:'single_parent_one_child',2:'single_parent_two_children'}.get(ch,'single_parent_three_plus_children')
    if pa==0 and ch>0: return 'children_only'
    return 'empty'

def minimum_cost(world,h):
    try: return max(0.0,num(world.needs_system.household_minimum_need_units(h),0.0))*max(EPS,num(world.household_planning_price_this_step(),1.0))
    except Exception: return np.nan

def current_events(world,hid):
    step=getattr(world,'current_step_index',0); out=[]
    for e in getattr(world,'household_lifecycle_events',[]):
        if e.get('step')==step and str(e.get('household_id'))==str(hid): out.append(e)
    for e in getattr(world,'household_lifecycle_transfer_events',[]):
        if e.get('week')==step and (str(e.get('source_household_id'))==str(hid) or str(e.get('destination_household_id'))==str(hid)): out.append(e)
    return out

def flow_components(h):
    wage=num(getattr(h,'wage_income_this_step',np.nan)); div=num(getattr(h,'dividend_income_this_step',0.0),0.0); pen=num(getattr(h,'pension_income_this_step',0.0),0.0)
    private=sum(num(getattr(h,n,0.0),0.0) for n in ('private_support_received_this_step','wealth_transfer_received_this_step','intergenerational_transfer_income_this_step'))
    total=num(getattr(h,'income_this_step',np.nan)); other=total-wage-div-pen-private if math.isfinite(total) and math.isfinite(wage) else np.nan
    return wage,div,pen,private,other,total

def age_prod(age):
    from productivity import age_productivity
    return float(age_productivity(age))
def capture_row(world,h,week):
    members=members_for(world,h); eligible=[p for p in members if bool(world.labor_formally_eligible(p))]; employed=[p for p in eligible if getattr(p,'firm_id',None) is not None]
    children=[p for p in members if num(p.age)<20]; elderly=[p for p in members if num(p.age)>=65]; adults=[p for p in members if 20<=num(p.age)<65]
    prod=[max(0.0,age_prod(p.age)) for p in employed]; wage,div,pen,private,other,total=flow_components(h); minimum=minimum_cost(world,h)
    desired=num(getattr(h,'desired_consumption_this_step',np.nan)); realized=num(getattr(h,'consumption_this_step',np.nan)); affordable=num(getattr(h,'affordable_consumption_this_step',np.nan)); cash=num(getattr(h,'wealth',np.nan)); opening=num(getattr(h,'wealth_before_income_this_step',np.nan))
    events=current_events(world,h.id); types=';'.join(sorted({str(e.get('event_type',e.get('type',''))) for e in events if e.get('event_type',e.get('type',''))}))
    formation=[e for e in events if any(t in str(e.get('event_type',e.get('type',''))).lower() for t in ('form','create','merge','marriage'))]
    formation_week=num(formation[0].get('step',formation[0].get('week',np.nan))) if formation else np.nan
    return {
      'household_id':h.id,'week':week,'runtime_step':int(getattr(world,'current_step_index',week-1)),'closing_cash':cash,'opening_cash':opening,'liquidity_weeks':ratio(cash,minimum),'minimum_consumption_cost':minimum,'household_size':len(members),'household_type':household_type(h,members),'adult_count':len(adults),'child_count':len(children),'elderly_count':len(elderly),'labor_eligible_count':len(eligible),'employed_member_count':len(employed),'nonemployed_labor_eligible_count':max(0,len(eligible)-len(employed)),
      'wage_income':wage,'dividend_income':div,'pension_income':pen,'private_transfer_income':private,'other_known_cash_income':other,'total_income':total,'desired_consumption':desired,'realized_consumption':realized,'income_to_need':ratio(total,minimum),'wage_to_need':ratio(wage,minimum),'earners_per_household_member':ratio(len(employed),len(members)),'need_per_earner':minimum/max(len(employed),1) if math.isfinite(minimum) else np.nan,'wage_per_earner':wage/max(len(employed),1) if math.isfinite(wage) else np.nan,'total_effective_labor':sum(prod),'effective_labor_per_earner':sum(prod)/max(len(employed),1) if employed else 0.0,'mean_worker_age':mean_or_nan([p.age for p in employed]),'mean_age_productivity':mean_or_nan(prod),'mean_realized_wage_per_earner':wage/max(len(employed),1) if employed and math.isfinite(wage) else np.nan,'dependency_ratio':(len(members)-len(employed))/max(len(employed),1),'cash_constraint_binding':bool(math.isfinite(desired) and math.isfinite(affordable) and desired>affordable+EPS),'suppressed_consumption_amount':max(0.0,desired-realized) if math.isfinite(desired) and math.isfinite(realized) else np.nan,'saving':num(getattr(h,'saving_this_step',np.nan)),'event_types':types,'formation_event_available':bool(formation),'formation_week':formation_week,'household_age_weeks':week-formation_week if math.isfinite(formation_week) else np.nan
    }, employed

def state_fingerprint(world):
    people=tuple(sorted((p.id,int(getattr(p,'age_weeks',0)),bool(getattr(p,'alive',False)),getattr(p,'household_id',None),getattr(p,'firm_id',None)) for p in getattr(world,'population',[])))
    households=tuple(sorted((h.id,round(num(getattr(h,'wealth',0.0),0.0),10),tuple(getattr(h,'parents',[])),tuple(getattr(h,'children',[]))) for h in getattr(world,'households',[])))
    firms=tuple(sorted((getattr(f,'firm_id',None),round(num(getattr(f,'cash',0.0),0.0),10),round(num(getattr(f,'inventory_units',0.0),0.0),10),tuple(getattr(f,'employee_ids',[]))) for f in world.operating_firms()))
    money=num(getattr(getattr(getattr(world,'firm_system',None),'central_bank',None),'money_supply',0.0),0.0)
    return hashlib.sha256(repr((people,households,firms,round(money,10),len(getattr(world,'population_history',[])))).encode()).hexdigest()

def run_probe(harness,instrument):
    world,_=harness.restore_world(); world.household_wealth_instrumentation_enabled=bool(instrument); world.household_employer_exposure_instrumentation_enabled=bool(instrument); world.steps=WINDOW
    rows=[]; worker_rows=[]; fps=[]
    for week in range(1,WINDOW+1):
        world.step(); fps.append(state_fingerprint(world))
        for h in list(world.households):
            if getattr(h,'settlement_only',False) or h.size()<=0: continue
            row,workers=capture_row(world,h,week); rows.append(row)
            for p in workers: worker_rows.append({'household_id':h.id,'week':week,'person_id':p.id,'age':p.age,'age_productivity':age_prod(p.age),'effective_labor':age_prod(p.age),'firm_id':getattr(p,'firm_id',None),'realized_wage':np.nan,'realized_wage_status':'not persisted per member; household wage aggregate only'})
    return world,pd.DataFrame(rows),pd.DataFrame(worker_rows),fps

def write_csv(name,frame):
    OUT.mkdir(parents=True,exist_ok=True); (frame if isinstance(frame,pd.DataFrame) else pd.DataFrame(frame)).to_csv(OUT/name,index=False,encoding='utf-8-sig')

def stats_rows(frame,col,metrics):
    if frame.empty: return pd.DataFrame([{'group':'UNAVAILABLE','status':'NO_ROWS'}])
    out=[]
    for group,q in frame.groupby(col,dropna=False,observed=False):
        row={'group':str(group),'observations':len(q),'households':q.household_id.nunique()}
        for metric in metrics:
            x=pd.to_numeric(q[metric],errors='coerce'); row['mean_'+metric]=x.mean(); row['median_'+metric]=x.median()
        out.append(row)
    return pd.DataFrame(out)
def event_detail(all_rows,events,name):
    if all_rows.empty or events.empty: return pd.DataFrame([{'event':name,'status':'NO_OBSERVED_EVENTS'}])
    events=events.sort_values(['household_id','week']).head(EVENT_LIMIT); out=[]; indexed=all_rows.set_index(['household_id','week'])
    for item in events.itertuples(index=False):
        for offset in (-1,0,1,4):
            key=(item.household_id,int(item.week+offset))
            if key not in indexed.index: continue
            row=indexed.loc[key]
            if isinstance(row,pd.DataFrame): row=row.iloc[0]
            out.append({'event':name,'household_id':item.household_id,'event_week':int(item.week),'event_relative_week':offset,**row.to_dict(),'event_window_note':'weekly short replay; no interpolation'})
    return pd.DataFrame(out)

def main():
    if not CHECKPOINT.exists(): raise FileNotFoundError(CHECKPOINT)
    if OUT.exists(): shutil.rmtree(OUT)
    harness=load_module(ROOT/'test/step15_mature_genealogy_private_support_experiment.py','step17_i_mature_harness')
    _,off_rows,_,off_fp=run_probe(harness,False); _,rows,member_rows,on_fp=run_probe(harness,True)
    if rows.empty: raise RuntimeError('short replay produced no household rows')
    rows=rows.sort_values(['household_id','week']).reset_index(drop=True)
    rows['liquidity_group']=pd.cut(pd.to_numeric(rows.liquidity_weeks,errors='coerce'),bins=[-np.inf,LOW,NONLOW,np.inf],labels=['DEEP_LOW','INTERMEDIATE','NON_LOW'],right=False)
    rows['low_now']=rows.liquidity_group=='DEEP_LOW'; rows['previous_group']=rows.groupby('household_id').liquidity_group.shift(1)
    rows['entry_deep_low']=rows.low_now & rows.previous_group.notna() & (rows.previous_group!='DEEP_LOW')
    rows['exit_deep_low']=(rows.liquidity_group!='DEEP_LOW') & (rows.previous_group=='DEEP_LOW')
    rows['accumulation_margin']=rows.total_income-rows.realized_consumption; rows['income_need_gap']=rows.total_income-rows.minimum_consumption_cost
    first=rows[rows.week==rows.week.min()].copy(); low_first=rows[rows.low_now].sort_values(['week','household_id']).groupby('household_id',as_index=False).head(1).sort_values(['week','household_id'])
    persistent_ids=set(); hp=HOUT/'persistent_low_households.csv'
    if hp.exists(): persistent_ids=set(pd.read_csv(hp,usecols=['household_id']).household_id.tolist())
    preferred=low_first[low_first.household_id.isin(persistent_ids)]; fallback=low_first[~low_first.household_id.isin(persistent_ids)]; low_targets=pd.concat([preferred,fallback],ignore_index=True).drop_duplicates('household_id').head(SAMPLE_LIMIT); low_ids=set(low_targets.household_id.tolist())
    controls=[]; used_controls=set()
    for low in low_targets.itertuples(index=False):
        pool=rows[rows.liquidity_group=='NON_LOW'].copy().sort_values(['household_id','week']).drop_duplicates('household_id')
        pool['week_distance']=(pool.week-low.week).abs(); pool['distance']=(pool.minimum_consumption_cost-low.minimum_consumption_cost).abs()
        pool['structure_distance']=(pool.household_size-low.household_size).abs()+(pool.elderly_count-low.elderly_count).abs()+(pool.employed_member_count-low.employed_member_count).abs()
        same=pool[pool.week_distance==0]; exact=same[same.structure_distance==0]
        if exact.empty: exact=same[same.household_size==low.household_size]
        if exact.empty: exact=same
        if exact.empty: exact=pool
        unused=exact[~exact.household_id.isin(used_controls)]
        if not unused.empty: exact=unused
        match=exact.sort_values(['week_distance','structure_distance','distance','household_id']).iloc[0]
        used_controls.add(match.household_id)
        controls.append({'low_household_id':low.household_id,'control_household_id':match.household_id,'match_week':low.week,'match_size':low.household_size,'match_elderly_count':low.elderly_count,'match_employed_count':low.employed_member_count,'need_distance':float(abs(match.minimum_consumption_cost-low.minimum_consumption_cost)),'matching_note':'same week; exact size/elderly/employed count first; nearest need; ID tie-break'})
    control_ids=set(x['control_household_id'] for x in controls)
    rows['sample_group']=np.where(rows.household_id.isin(low_ids),'DEEP_LOW_TARGET',np.where(rows.household_id.isin(control_ids),'MATCHED_NON_LOW_CONTROL','UNSELECTED'))
    selected=rows[rows.sample_group!='UNSELECTED'].copy(); member_rows['sample_group']=np.where(member_rows.household_id.isin(low_ids),'DEEP_LOW_TARGET',np.where(member_rows.household_id.isin(control_ids),'MATCHED_NON_LOW_CONTROL','UNSELECTED'))
    detail_cols=['household_id','week','closing_cash','opening_cash','liquidity_weeks','minimum_consumption_cost','household_size','household_type','adult_count','child_count','elderly_count','labor_eligible_count','employed_member_count','nonemployed_labor_eligible_count','wage_income','dividend_income','pension_income','private_transfer_income','other_known_cash_income','total_income','desired_consumption','realized_consumption','cash_constraint_binding','suppressed_consumption_amount','saving','income_to_need','wage_to_need','earners_per_household_member','need_per_earner','wage_per_earner','total_effective_labor','effective_labor_per_earner','mean_worker_age','mean_age_productivity','mean_realized_wage_per_earner','dependency_ratio','event_types','formation_event_available','formation_week','household_age_weeks','sample_group']
    write_csv('sampled_household_week_detail.csv',selected[detail_cols]); write_csv('member_labor_aggregate.csv',member_rows[member_rows.sample_group!='UNSELECTED'])
    match_rows=[]
    for item in controls:
        low=selected[selected.household_id==item['low_household_id']]; ctl=selected[selected.household_id==item['control_household_id']]; result=dict(item)
        for label,frame in (('low',low),('control',ctl)):
            for metric in ('income_to_need','wage_to_need','employed_member_count','nonemployed_labor_eligible_count','mean_age_productivity','effective_labor_per_earner','mean_realized_wage_per_earner','need_per_earner','dependency_ratio','minimum_consumption_cost','total_income','realized_consumption','saving'):
                result[label+'_mean_'+metric]=pd.to_numeric(frame[metric],errors='coerce').mean() if not frame.empty else np.nan
        match_rows.append(result)
    write_csv('low_vs_matched_comparison.csv',match_rows)
    write_csv('labor_supply_gap.csv',stats_rows(selected,'sample_group',['employed_member_count','labor_eligible_count','nonemployed_labor_eligible_count','earners_per_household_member','income_to_need']))
    employed_selected=selected[selected.employed_member_count>0]; write_csv('productivity_gap.csv',stats_rows(employed_selected,'sample_group',['mean_worker_age','mean_age_productivity','effective_labor_per_earner','mean_realized_wage_per_earner','wage_to_need']))
    selected['elderly_employment_stratum']=np.select([selected.elderly_count>0,selected.employed_member_count>0],['ELDERLY_WITH_EMPLOYED','NON_ELDERLY_OR_NO_EARNER'],default='NON_ELDERLY_NO_EMPLOYED')
    selected.loc[(selected.elderly_count>0)&(selected.employed_member_count==0),'elderly_employment_stratum']='ELDERLY_NO_EMPLOYED'
    selected.loc[(selected.elderly_count==0)&(selected.employed_member_count>0),'elderly_employment_stratum']='NON_ELDERLY_WITH_EMPLOYED'
    selected.loc[(selected.elderly_count==0)&(selected.employed_member_count==0),'elderly_employment_stratum']='NON_ELDERLY_NO_EMPLOYED'
    write_csv('elderly_income_gap.csv',stats_rows(selected,'elderly_employment_stratum',['income_to_need','wage_to_need','minimum_consumption_cost','total_income','wage_income']))
    selected['dependent_burden_stratum']=np.select([selected.dependency_ratio>=2,selected.dependency_ratio>=1],['HIGH_DEPENDENCY_RATIO_GE_2','DEPENDENCY_RATIO_1_TO_2'],default='DEPENDENCY_RATIO_LT_1')
    write_csv('dependent_burden_gap.csv',stats_rows(selected,'dependent_burden_stratum',['child_count','elderly_count','household_size','need_per_earner','dependency_ratio','income_to_need']))
    all_active=rows[rows.liquidity_group.notna()]; type_rows=[]; all_low=max(1,int((all_active.liquidity_group=='DEEP_LOW').sum()))
    for kind,q in all_active.groupby('household_type',dropna=False,observed=False):
        lowq=q[q.liquidity_group=='DEEP_LOW']; type_rows.append({'household_type':str(kind),'all_household_weeks':len(q),'all_households':q.household_id.nunique(),'low_household_weeks':len(lowq),'low_households':lowq.household_id.nunique(),'low_rate_within_type':ratio(len(lowq),len(q)),'low_share_of_all_low':ratio(len(lowq),all_low),'source_note':'26-week live mature-state replay'})
    write_csv('household_type_low_tail.csv',type_rows)
    life=selected[['household_id','week','sample_group','event_types','formation_event_available','formation_week','household_age_weeks']].copy(); life['formation_status']=np.where(life.formation_event_available,'AUTHORITATIVE_EVENT_OBSERVED','NO_EVENT_IN_SHORT_WINDOW'); write_csv('lifecycle_formation_audit.csv',life)
    source_rows=[]
    for group,q in selected.groupby('sample_group',observed=False):
        for name in ('wage_income','dividend_income','pension_income','private_transfer_income','other_known_cash_income'):
            value=pd.to_numeric(q[name],errors='coerce'); total=float(value.sum()); known=sum(float(pd.to_numeric(q[x],errors='coerce').sum()) for x in ('wage_income','dividend_income','pension_income','private_transfer_income','other_known_cash_income'))
            source_rows.append({'sample_group':group,'income_source':name,'observations':len(q),'mean_source_income':value.mean(),'total_source_income':total,'share_of_known_income':total/known if abs(known)>EPS else np.nan,'structural_zero_in_control':bool(group=='MATCHED_NON_LOW_CONTROL' and abs(total)<=EPS and name in ('dividend_income','pension_income','private_transfer_income')),'source_note':'runtime Household fields; other is known-income residual'})
    write_csv('income_source_composition.csv',source_rows)
    write_csv('low_tail_entry_event_detail.csv',event_detail(rows,rows[rows.entry_deep_low],'ENTRY_INTO_DEEP_LOW')); write_csv('low_tail_exit_event_detail.csv',event_detail(rows,rows[rows.exit_deep_low],'EXIT_FROM_DEEP_LOW'))
    means=selected.groupby(['household_id','sample_group'],as_index=False).agg(mean_income_need=('income_to_need','mean'),mean_wage_need=('wage_to_need','mean'),mean_employed=('employed_member_count','mean'),mean_nonemployed_eligible=('nonemployed_labor_eligible_count','mean'),mean_age_productivity=('mean_age_productivity','mean'),mean_wage_per_earner=('mean_realized_wage_per_earner','mean'),mean_need_per_earner=('need_per_earner','mean'),mean_dependency=('dependency_ratio','mean'),mean_elderly=('elderly_count','mean'),formation_observed=('formation_event_available','max'))
    lowmean=means[means.sample_group=='DEEP_LOW_TARGET']; cref={int(x.household_id):x for x in means[means.sample_group=='MATCHED_NON_LOW_CONTROL'].itertuples(index=False)}; mechanism=[]
    for item in lowmean.itertuples(index=False):
        pair=[x for x in controls if x['low_household_id']==item.household_id]; ctl=cref.get(int(pair[0]['control_household_id'])) if pair else None
        no_earner=item.mean_employed<=EPS; lowprod=bool(ctl is not None and item.mean_age_productivity < num(getattr(ctl,'mean_age_productivity',np.nan))-EPS and item.mean_wage_per_earner < num(getattr(ctl,'mean_wage_per_earner',np.nan))-EPS); elderly_gap=item.mean_elderly>EPS and item.mean_income_need<1; dependent=bool(ctl is not None and (item.mean_dependency>num(getattr(ctl,'mean_dependency',np.nan))+EPS or item.mean_need_per_earner>num(getattr(ctl,'mean_need_per_earner',np.nan))+EPS)); formation=bool(item.formation_observed)
        flags={'NO_EARNER_OR_LABOR_SUPPLY_GAP':no_earner,'LOW_PRODUCTIVITY_WAGE_GAP':lowprod,'ELDERLY_INCOME_GAP':elderly_gap,'HIGH_DEPENDENT_BURDEN':dependent,'HOUSEHOLD_FORMATION_RESET':formation}; active=[k for k,v in flags.items() if v]; label='MIXED' if len(active)>1 else (active[0] if active else 'UNCLASSIFIED'); mechanism.append({'household_id':item.household_id,**flags,'active_mechanisms':';'.join(active),'primary_label':label,'rule_note':'sample rule; low productivity requires lower matched age-productivity and wage-per-earner'})
    mech=pd.DataFrame(mechanism); write_csv('mechanism_overlap.csv',mech)
    if mech.empty: primary=pd.DataFrame([{'primary_label':'UNCLASSIFIED','households':0,'share':np.nan}])
    else: primary=mech.primary_label.value_counts().rename_axis('primary_label').reset_index(name='households'); primary['share']=primary.households/len(mech)
    write_csv('mechanism_primary_classification.csv',mech); write_csv('mechanism_contribution_summary.csv',primary)
    model_data=rows[['low_now','employed_member_count','household_size','elderly_count','child_count','mean_age_productivity','mean_realized_wage_per_earner','need_per_earner','dependency_ratio']].copy(); model_data=model_data.rename(columns={'low_now':'target','mean_age_productivity':'age_productivity','mean_realized_wage_per_earner':'wage_per_earner'}); model_data.target=model_data.target.astype(int)
    predictors=['employed_member_count','household_size','elderly_count','child_count','age_productivity','wage_per_earner','need_per_earner','dependency_ratio']; model_rows=[]
    try:
        from sklearn.linear_model import LogisticRegression
        x=model_data[predictors].replace([np.inf,-np.inf],np.nan); x=x.fillna(x.median(numeric_only=True)).fillna(0).to_numpy(float); y=model_data.target.to_numpy(int)
        if len(np.unique(y))<2: raise ValueError('single target class')
        scale=x.std(axis=0); scale[scale<EPS]=1; z=(x-x.mean(axis=0))/scale; fit=LogisticRegression(max_iter=1000,random_state=0).fit(z,y)
        for name,coef in zip(predictors,fit.coef_[0]): model_rows.append({'predictor':name,'coefficient_standardized':float(coef),'odds_ratio_per_sd':float(np.exp(coef)),'target':'P(DEEP_LOW at observed week)','model':'descriptive household-level logistic; not causal'})
        model_rows.append({'predictor':'MODEL_INTERCEPT','coefficient_standardized':float(fit.intercept_[0]),'odds_ratio_per_sd':np.nan,'target':'P(DEEP_LOW at observed week)','model':'descriptive; standardized predictors'})
    except Exception as exc: model_rows.append({'predictor':'UNAVAILABLE','coefficient_standardized':np.nan,'odds_ratio_per_sd':np.nan,'target':'P(DEEP_LOW at observed week)','model':'not fit: '+type(exc).__name__})
    write_csv('descriptive_logistic_model.csv',model_rows)
    parity=[{'week':i,'state_fingerprint_off':a,'state_fingerprint_on':b,'state_equal':a==b,'rng_status':'same restored checkpoint; observer made no RNG calls'} for i,(a,b) in enumerate(zip(off_fp,on_fp),1)]; write_csv('observability_parity.csv',parity)
    if mech.empty: verdict='G. OBSERVABILITY_STILL_INSUFFICIENT'; dominant='UNCLASSIFIED'
    else:
        pc=mech.primary_label.value_counts(); dominant=str(pc.index[0]); verdict='F. LOW_TAIL_HAS_MULTIPLE_OVERLAPPING_INCOME_MECHANISMS' if len(pc)>1 and pc.iloc[0]<len(mech) else {'NO_EARNER_OR_LABOR_SUPPLY_GAP':'A. LOW_TAIL_PRIMARILY_LABOR_SUPPLY_GAP','LOW_PRODUCTIVITY_WAGE_GAP':'B. LOW_TAIL_PRIMARILY_LOW_PRODUCTIVITY_WAGE_GAP','ELDERLY_INCOME_GAP':'C. LOW_TAIL_PRIMARILY_ELDERLY_INCOME_GAP','HIGH_DEPENDENT_BURDEN':'D. LOW_TAIL_PRIMARILY_DEPENDENT_BURDEN','HOUSEHOLD_FORMATION_RESET':'E. LOW_TAIL_PRIMARILY_HOUSEHOLD_FORMATION_RESET'}.get(dominant,'G. OBSERVABILITY_STILL_INSUFFICIENT')
    flags={'verdict':verdict,'diagnostic_horizon_weeks':WINDOW,'checkpoint':str(CHECKPOINT),'checkpoint_semantics':'accepted mature population checkpoint rooted at absolute week 2600; fresh economic state restored by mature harness','sample_low_targets':len(low_ids),'sample_matched_controls':len(control_ids),'live_household_week_rows':len(rows),'short_replay_low_rows':int((rows.liquidity_group=='DEEP_LOW').sum()),'member_level_fields_authoritative':True,'member_realized_wage_authoritative':False,'formation_events_authoritative_if_observed':True,'no_long_run':True,'economic_behavior_changed':False,'new_rng_draws':0,'parity_all_weeks':bool(parity) and all(x['state_equal'] for x in parity),'no_policy_implementation':True,'no_payg_activation':True,'no_retirement_or_wage_change':True,'step17_j_started':False,'income_source_note':'CONTROL has no active PAYG/private policy; zero components are runtime fields, not fabricated history','unavailable_note':'per-member realized wage allocation is unavailable; household wage per earner and Person age/productivity are available'}
    (OUT/'acceptance_flags.json').write_text(json.dumps(flags,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    summary=['# Step17.I Acceptance Summary','','Verdict: '+verdict,'',f'{WINDOW}-week short replay from the accepted mature population checkpoint rooted at absolute week 2600; no 5000-week run. Live rows: {len(rows):,}; sampled DEEP_LOW targets: {len(low_ids)}; deterministic matched controls: {len(control_ids)}.','', 'Authoritative runtime fields include member counts, labor eligibility, employed members, age, age-productivity, effective labor, household income components, consumption, need, cash constraint and lifecycle events. Per-member realized wage allocation was not persisted; it remains unavailable, while household wage-per-earner is an aggregate calculation.','',f'Dominant sample label: {dominant}. Labels are descriptive; low productivity requires lower matched age-productivity and wage-per-earner, while elderly/dependent labels are overlap indicators, not causal claims.','',f"Observability parity: {'PASS' if flags['parity_all_weeks'] else 'FAIL'} across {len(parity)} weeks. No policy, retirement, wage, RNG or economic behavior change; Step17.J not started.",'','Formation and event conclusions are limited to the 26-week replay. Missing events are not reconstructed from IDs.','', 'Artifacts: test/output/step17_i_low_income_mechanism_attribution/']
    (OUT/'acceptance_summary.md').write_text('\n'.join(summary)+'\n',encoding='utf-8')

if __name__=='__main__': main()
