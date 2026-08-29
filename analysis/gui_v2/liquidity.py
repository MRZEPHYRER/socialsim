"""Step17.F household liquidity analysis and embedded GUI."""
from pathlib import Path
import math,numpy as np,pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox,QHBoxLayout,QLabel,QPushButton,QSlider,QVBoxLayout,QWidget
from analysis.gui_v2.widgets import ChartCanvas,DataTable,MetricCard,Notice
BUCKET_LABELS=("<0.25","0.25-0.5","0.5-1","1-2","2-4",">=4")
GROUP_LABELS={"ALL":"ALL HOUSEHOLDS","ELDERLY":"ELDERLY HOUSEHOLDS","NON_ELDERLY":"NON-ELDERLY HOUSEHOLDS","EMPLOYED":"EMPLOYED-MEMBER HOUSEHOLDS","PENSION_RECIPIENTS":"PENSION RECIPIENTS (THIS WEEK)","PRIVATE_TRANSFER_RECIPIENTS":"PRIVATE TRANSFER RECIPIENTS (THIS WEEK)","PAYG_CONTRIBUTORS":"PAYG CONTRIBUTORS (THIS WEEK)","GENEALOGY_UNCOVERED_ELDERLY":"GENEALOGY-UNCOVERED ELDERLY"}
def load_liquidity_snapshot(run_dir):
 root=Path(run_dir)
 p=root/"household_liquidity_weekly_snapshot.csv"
 g=root/"long_horizon_household_snapshots.csv"
 if g.exists():
  f=pd.read_csv(g,low_memory=False)
  f=f.rename(columns={"weekly_minimum_consumption_cost":"minimum_consumption_cost","consumption":"realized_consumption"})
  f["branch"]="CONTROL"
  f["settlement_only"]=False
 elif p.exists():
  f=pd.read_csv(p,low_memory=False)
 else:return pd.DataFrame()
 req={"global_step","branch","household_id","closing_cash","minimum_consumption_cost","liquidity_weeks"}
 if not req.issubset(f):return pd.DataFrame()
 for c in ("global_step","household_id","closing_cash","minimum_consumption_cost","liquidity_weeks"):f[c]=pd.to_numeric(f[c],errors="coerce")
 for c in ("settlement_only","elderly_household","employed_member","genealogy_uncovered_elderly","pension_recipient_this_week","private_transfer_recipient_this_week","payg_contributor_this_week"):f[c]=f[c].fillna(False).astype(bool) if c in f else False
 return f.sort_values(["branch","global_step","household_id"]).reset_index(drop=True)
def group_filter(f,g):
 m={"ELDERLY":"elderly_household","NON_ELDERLY":"elderly_household","EMPLOYED":"employed_member","PENSION_RECIPIENTS":"pension_recipient_this_week","PRIVATE_TRANSFER_RECIPIENTS":"private_transfer_recipient_this_week","PAYG_CONTRIBUTORS":"payg_contributor_this_week","GENEALOGY_UNCOVERED_ELDERLY":"genealogy_uncovered_elderly"}
 if g not in m:return f
 return f[f[m[g]]] if g!="NON_ELDERLY" else f[~f[m[g]]]
def gini(v):
 a=np.sort(np.maximum(pd.to_numeric(pd.Series(v),errors="coerce").fillna(0).to_numpy(float),0))
 return 0.0 if len(a)==0 or a.sum()<=1e-12 else float(np.sum((2*np.arange(1,len(a)+1)-len(a)-1)*a)/(len(a)*a.sum()))
def distribution_metrics(f,value="liquidity_weeks"):
 v=pd.to_numeric(f.get(value,pd.Series(dtype=float)),errors="coerce").dropna();c=pd.to_numeric(f.get("closing_cash",pd.Series(dtype=float)),errors="coerce").reindex(v.index).fillna(0).clip(lower=0)
 keys=("count","p_lt_025","p_lt_05","p_lt_1","p_lt_2","median","p10","p25","p75","p90","cash_gini","bottom50_cash_share","top10_cash_share","top1_cash_share")
 if v.empty:return {k:np.nan for k in keys}
 a=np.sort(c.to_numpy());t=a.sum();share=lambda i:float(a[i:].sum()/t) if t>1e-12 else 0.0
 return {"count":len(v),"p_lt_025":float((v<.25).mean()),"p_lt_05":float((v<.5).mean()),"p_lt_1":float((v<1).mean()),"p_lt_2":float((v<2).mean()),"median":float(v.quantile(.5)),"p10":float(v.quantile(.1)),"p25":float(v.quantile(.25)),"p75":float(v.quantile(.75)),"p90":float(v.quantile(.9)),"cash_gini":gini(c),"bottom50_cash_share":float(a[:max(1,len(a)//2)].sum()/t) if t>1e-12 else 0.0,"top10_cash_share":share(max(0,len(a)-max(1,math.ceil(len(a)*.1)))),"top1_cash_share":share(max(0,len(a)-max(1,math.ceil(len(a)*.01))))}
def bucket_for(v):
 if pd.isna(v):return "UNAVAILABLE"
 return BUCKET_LABELS[0] if v<.25 else BUCKET_LABELS[1] if v<.5 else BUCKET_LABELS[2] if v<1 else BUCKET_LABELS[3] if v<2 else BUCKET_LABELS[4] if v<4 else BUCKET_LABELS[5]
def bucket_summary(f):
 s=f.get("liquidity_weeks",pd.Series(dtype=float)).map(bucket_for);n=len(f)
 return pd.DataFrame([{"bucket":b,"count":int((s==b).sum()),"share":float((s==b).sum()/n) if n else np.nan} for b in BUCKET_LABELS])
class HouseholdLiquidityExplorer(QWidget):
 def __init__(self,run_dir,language="zh",parent=None):
  super().__init__(parent);self.frame=load_liquidity_snapshot(run_dir);self._build()
 def _build(self):
  self.weeks=sorted(self.frame.global_step.dropna().astype(int).unique()) if not self.frame.empty else [0];self.week=QSlider(Qt.Orientation.Horizontal);self.week.setRange(0,max(0,len(self.weeks)-1));self.week_label=QLabel()
  self.branch=QComboBox();[self.branch.addItem(b,b) for b in sorted(self.frame.branch.dropna().astype(str).unique())];i=self.branch.findData("CONTROL");self.branch.setCurrentIndex(i if i>=0 else 0)
  self.group=QComboBox();[self.group.addItem(v,k) for k,v in GROUP_LABELS.items()];self.variable=QComboBox();self.variable.addItem("liquidity_weeks","liquidity_weeks");self.variable.addItem("Household cash","closing_cash")
  self.histogram=ChartCanvas();self.ecdf=ChartCanvas();self.buckets=DataTable(pd.DataFrame(),"en");self.inspect=DataTable(pd.DataFrame(),"en");self.cards=QHBoxLayout();self.delta=QLabel()
  prev=QPushButton("Previous snapshot");nxt=QPushButton("Next snapshot");b13=QPushButton("-13 weeks");f13=QPushButton("+13 weeks");b52=QPushButton("-52 weeks");f52=QPushButton("+52 weeks");b260=QPushButton("-260 weeks");f260=QPushButton("+260 weeks");r=QHBoxLayout();r.addWidget(QLabel("Snapshot week"));r.addWidget(self.week,1);r.addWidget(self.week_label);[r.addWidget(x) for x in (prev,nxt,b13,f13,b52,f52,b260,f260)]
  q=QHBoxLayout();[q.addWidget(x) for x in (QLabel("Branch"),self.branch,QLabel("Group"),self.group,QLabel("X"),self.variable)]
  root=QVBoxLayout(self);root.addWidget(Notice("用于人工检查低流动性家庭是否持续停留在近零区域，以及政策分支是否改变进入和退出。"));root.addLayout(r);root.addLayout(q);root.addWidget(QLabel("事件组是本周事件；老年和就业是状态组。"));root.addWidget(self.delta);root.addLayout(self.cards);root.addWidget(self.histogram);root.addWidget(self.ecdf);root.addWidget(QLabel("Low-liquidity buckets"));root.addWidget(self.buckets);root.addWidget(QLabel("Representative households"));root.addWidget(self.inspect)
  self.week.valueChanged.connect(self._refresh);self.branch.currentIndexChanged.connect(self._refresh);self.group.currentIndexChanged.connect(self._refresh);self.variable.currentIndexChanged.connect(self._refresh);prev.clicked.connect(lambda:self.week.setValue(max(0,self.week.value()-1)));nxt.clicked.connect(lambda:self.week.setValue(min(self.week.maximum(),self.week.value()+1)));b13.clicked.connect(lambda:self._move_week(-13));f13.clicked.connect(lambda:self._move_week(13));b52.clicked.connect(lambda:self._move_week(-52));f52.clicked.connect(lambda:self._move_week(52));b260.clicked.connect(lambda:self._move_week(-260));f260.clicked.connect(lambda:self._move_week(260));self._refresh()
 def _move_week(self,delta):
  if not self.weeks:return
  target=self.weeks[self.week.value()]+delta
  index=min(range(len(self.weeks)),key=lambda i:abs(self.weeks[i]-target))
  self.week.setValue(index)
 def _refresh(self,*_):
  w=self.weeks[self.week.value()];f=group_filter(self.frame[(self.frame.branch.astype(str)==str(self.branch.currentData()))&(self.frame.global_step==w)],self.group.currentData());v=self.variable.currentData();m=distribution_metrics(f,v);self.week_label.setText(str(w))
  while self.cards.count():self.cards.takeAt(0).widget().deleteLater()
  for title,key in (("Households","count"),("P(<.25)","p_lt_025"),("P(<.5)","p_lt_05"),("P(<1)","p_lt_1"),("Median","median"),("Cash Gini","cash_gini")):self.cards.addWidget(MetricCard(title,m[key]))
  x=pd.to_numeric(f.get(v,pd.Series(dtype=float)),errors="coerce").dropna();self.histogram.figure.clear();a=self.histogram.figure.add_subplot(111)
  if x.empty:a.text(.5,.5,"No observations",ha="center",va="center",transform=a.transAxes)
  else:a.hist(x.clip(upper=4) if v=="liquidity_weeks" else x,bins=[0,.25,.5,1,2,4] if v=="liquidity_weeks" else 24,color="#007f7b",edgecolor="white");a.set_xlim(0,4) if v=="liquidity_weeks" else None
  if v=="liquidity_weeks":[a.axvline(z,color="#d17b24",ls="--",lw=.8) for z in (.25,.5,1,2)]
  self.histogram.draw_idle();self.ecdf.figure.clear();a=self.ecdf.figure.add_subplot(111)
  if x.empty:a.text(.5,.5,"No observations",ha="center",va="center",transform=a.transAxes)
  else:y=np.sort(x.to_numpy());a.step(y,np.arange(1,len(y)+1)/len(y),where="post",color="#3178b8")
  if v=="liquidity_weeks":a.set_xlim(0,4);[a.axvline(z,color="#d17b24",ls="--",lw=.8) for z in (.25,.5,1,2)]
  a.set_title("ECDF",loc="left");self.ecdf.draw_idle();self.buckets.set_frame(bucket_summary(f),"en");q=f.assign(bucket=f.liquidity_weeks.map(bucket_for)).query("bucket!='UNAVAILABLE'").head(10);cols=[c for c in ("household_id","closing_cash","liquidity_weeks","elderly_household","employed_member","pension_recipient_this_week","private_transfer_recipient_this_week","payg_contributor_this_week") if c in q];self.inspect.set_frame(q[cols],"en")
  if str(self.branch.currentData())=="CONTROL":self.delta.setText("Control baseline selected")
  else:
   c=group_filter(self.frame[(self.frame.branch.astype(str)=="CONTROL")&(self.frame.global_step==w)],self.group.currentData());cm=distribution_metrics(c,v);self.delta.setText("Delta treatment - CONTROL (negative low-tail share is improvement): "+" ".join(f"{k}={m[k]-cm[k]:.4f}" for k in ("p_lt_025","p_lt_05","p_lt_1","median","cash_gini")))
__all__=["BUCKET_LABELS","GROUP_LABELS","load_liquidity_snapshot","group_filter","gini","distribution_metrics","bucket_for","bucket_summary","HouseholdLiquidityExplorer"]
