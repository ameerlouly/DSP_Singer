"""Step 5 SNR tab, independent of the spectrogram and calculation backend."""
import copy
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sounddevice as sd
import soundfile as sf
import step5_snr as snr


class SNRPanel(ttk.Frame):
    def __init__(self,parent,get_review):
        super().__init__(parent,padding=12)
        self.get_review=get_review;self.results=[];self.snapshot=None;self.busy=False;self.work=queue.Queue();self.directory=None
        self.threshold=tk.StringVar(value='20');self.reviewer=tk.StringVar();self.reason=tk.StringVar()
        self.info=tk.StringVar(value='Approve clips in Step 4, then set a threshold and calculate.')
        settings=ttk.LabelFrame(self,text='SNR settings',padding=10);settings.pack(fill='x')
        ttk.Label(settings,text='Clean threshold (dB):').pack(side='left')
        ttk.Entry(settings,textvariable=self.threshold,width=8).pack(side='left',padx=5)
        self.button=ttk.Button(settings,text='Calculate SNR',command=self.start);self.button.pack(side='left',padx=10)
        ttk.Button(settings,text='Open SNR review',command=self.open_review).pack(side='left',padx=4)
        ttk.Label(settings,text='Default: 20 dB. Energy method; WADA-SNR will be added separately.').pack(side='left')
        ttk.Label(self,textvariable=self.info,wraplength=1100).pack(fill='x',pady=8)
        frame=ttk.Frame(self);frame.pack(fill='both',expand=True)
        cols=('id','source','type','snr','auto','quality','note')
        self.table=ttk.Treeview(frame,columns=cols,show='headings',selectmode='extended')
        for col,title,width in [('id','Clip',120),('source','Source time (s)',150),('type','Speech / mixed',100),('snr','Energy SNR (dB)',120),('auto','Automatic',90),('quality','Current quality',100),('note','Notes',500)]:
            self.table.heading(col,text=title);self.table.column(col,width=width,stretch=col=='note')
        self.table.pack(side='left',fill='both',expand=True)
        scroll=ttk.Scrollbar(frame,orient='vertical',command=self.table.yview);scroll.pack(side='right',fill='y');self.table.configure(yscrollcommand=scroll.set)
        ttk.Label(self,text='Ctrl-click individual clips or Shift-click a range. Select one clip to play.').pack(anchor='w',pady=5)
        actions=ttk.Frame(self);actions.pack(fill='x')
        ttk.Label(actions,text='Reviewer:').pack(side='left');ttk.Entry(actions,textvariable=self.reviewer,width=14).pack(side='left',padx=4)
        ttk.Label(actions,text='Reason:').pack(side='left');ttk.Entry(actions,textvariable=self.reason,width=24).pack(side='left',padx=4)
        ttk.Button(actions,text='Mark selected clean',command=lambda:self.override('clean')).pack(side='left',padx=3)
        ttk.Button(actions,text='Mark selected noisy',command=lambda:self.override('noisy')).pack(side='left',padx=3)
        ttk.Button(actions,text='Restore automatic',command=lambda:self.override(None)).pack(side='left',padx=3)
        ttk.Button(actions,text='Play selected',command=self.play).pack(side='left',padx=3)
        ttk.Button(actions,text='Stop',command=sd.stop).pack(side='left',padx=3)
        ttk.Button(actions,text='Save updated report',command=self.save).pack(side='left',padx=3)
        self.after(100,self.poll)

    def start(self):
        if self.busy:return
        panel=self.get_review()
        if panel is None or not panel.winfo_exists():
            messagebox.showerror('Step 4 required','Open or create the Step 4 review first.',parent=self);return
        try:
            threshold=float(self.threshold.get())
            if not -50<=threshold<=100:raise ValueError('Enter a threshold between -50 and 100 dB.')
            snapshot=copy.deepcopy(panel.data)
            records=snr.session_records(snapshot,panel.source)
            if not records:raise ValueError('Approve at least one clip in Step 4 first.')
            directory=filedialog.askdirectory(title='Choose folder for the SNR HTML report',parent=self)
            if not directory:return
        except Exception as exc:messagebox.showerror('Cannot calculate SNR',str(exc),parent=self);return
        self.busy=True;self.button.configure(state='disabled');self.info.set('Calculating SNR...')
        def worker():
            try:
                result=snr.calculate(records,threshold,lambda n,total:self.work.put(('progress',(n,total))))
                self.work.put(('done',(result,threshold,directory,snapshot)))
            except Exception as exc:self.work.put(('error',str(exc)))
        threading.Thread(target=worker,daemon=True).start()

    def poll(self):
        try:
            while True:
                kind,payload=self.work.get_nowait()
                if kind=='progress':self.info.set(f'Calculating SNR: {payload[0]} / {payload[1]}');continue
                self.busy=False;self.button.configure(state='normal')
                if kind=='error':messagebox.showerror('SNR calculation failed',payload,parent=self);continue
                records,threshold,directory,snapshot=payload
                panel=self.get_review()
                if panel is None or not panel.winfo_exists() or panel.data!=snapshot:
                    messagebox.showwarning('Clips changed','Step 4 changed during calculation. Calculate again.',parent=self);continue
                self.results=records;self.snapshot=snapshot;self.applied_threshold=threshold;self.directory=directory
                self.refresh();self.save()
        except queue.Empty:pass
        if self.winfo_exists():self.after(100,self.poll)

    def check_current(self):
        panel=self.get_review()
        if not self.results or panel is None or not panel.winfo_exists() or panel.data!=self.snapshot:
            raise ValueError('Step 4 clips changed or no result is loaded. Calculate SNR again.')

    def open_review(self):
        import json
        from pathlib import Path
        try:
            panel=self.get_review()
            if panel is None or not panel.winfo_exists():raise ValueError('Open the matching Step 4 review first.')
            path=filedialog.askopenfilename(title='Open saved Step 5 SNR JSON',filetypes=[('SNR JSON','*.json')],parent=self)
            if not path:return
            payload=json.loads(Path(path).read_text(encoding='utf-8-sig'))
            if payload.get('format')!='speech_corpus_snr_v1':raise ValueError('Not a supported Step 5 SNR review.')
            records=payload['segments'];current=snr.session_records(panel.data,panel.source)
            keys=('id','path','read_start','read_end','source_start','source_end','content_label','synthetic_start','synthetic_end')
            if len(records)!=len(current) or any(any(a.get(k)!=b.get(k) for k in keys) for a,b in zip(records,current)):
                raise ValueError('The saved SNR results do not match the currently approved Step 4 clips. Recalculate SNR.')
            self.results=records;self.snapshot=copy.deepcopy(panel.data);self.applied_threshold=float(payload['threshold_db'])
            self.threshold.set(str(self.applied_threshold));self.directory=str(Path(path).parent);self.refresh()
            self.info.set(f'Loaded {len(records)} SNR results from {path}.')
        except Exception as exc:messagebox.showerror('Could not open SNR review',str(exc),parent=self)

    def refresh(self):
        self.table.delete(*self.table.get_children())
        for i,r in enumerate(self.results):
            score='N/A' if r['snr_db'] is None else f"{r['snr_db']:.2f}"
            self.table.insert('', 'end',iid=str(i),values=(r['id'],f"{r['source_start']:.2f} - {r['source_end']:.2f}",r['content_label'],score,r['automatic_quality'],r['quality'],r['note']))

    def override(self,label):
        try:
            self.check_current(); selected=self.table.selection()
            if not selected:raise ValueError('Select one or more clips first.')
            who=self.reviewer.get().strip();why=self.reason.get().strip()
            if not who or not why:raise ValueError('Enter reviewer name and reason for the change.')
            from datetime import datetime,timezone
            for row in selected:
                r=self.results[int(row)];before=r['quality'];after=r['automatic_quality'] if label is None else label
                r.setdefault('quality_changes',[]).append({'time_utc':datetime.now(timezone.utc).isoformat(),'reviewer':who,'reason':why,'before':before,'after':after})
                r.update(quality=after,override_by=who,override_reason=why)
            self.refresh();self.info.set(f'Updated {len(selected)} clips. Save the updated report.')
        except Exception as exc:messagebox.showerror('Cannot change label',str(exc),parent=self)

    def play(self):
        try:
            self.check_current();selected=self.table.selection()
            if len(selected)!=1:raise ValueError('Select exactly one clip to play.')
            r=self.results[int(selected[0])];sd.stop()
            data,rate=sf.read(r['path'],start=round(r['read_start']*16000),stop=round(r['read_end']*16000),dtype='float32')
            sd.play(data,rate)
        except Exception as exc:messagebox.showerror('Playback failed',str(exc),parent=self)

    def save(self):
        try:
            self.check_current();report,json_path,csv_path=snr.export_report(self.results,self.applied_threshold,self.directory)
            self.info.set(f'SNR report saved: {report} | JSON and CSV saved alongside it.')
        except Exception as exc:messagebox.showerror('Report not saved',str(exc),parent=self)
