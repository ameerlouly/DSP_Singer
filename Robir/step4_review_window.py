"""Step 4 review window; backend and export are in step4_review.py."""
from pathlib import Path
import tkinter as tk
import queue
import threading
from tkinter import ttk, filedialog, messagebox
import soundfile as sf
import sounddevice as sd
import step4_review as review


class _ReviewBehavior:
    def setup_review(self, source_wav, metadata_path, reviewer=''):
        self.source=Path(source_wav).resolve()
        try:
            self.data=review.load_session(metadata_path,self.source)
        except Exception:
            self.destroy()
            raise
        self.dirty=False
        self._form_selected_iid=None
        self._playback_stop = threading.Event()
        self._playback_thread = None
        self._playback_errors = queue.Queue()
        self._reseg_queue = queue.Queue()
        self._reseg_busy = False
        self._reseg_dialog = None
        self.reviewer=tk.StringVar(value=reviewer)
        self.label=tk.StringVar(value='speech')
        self.start_var=tk.StringVar()
        self.end_var=tk.StringVar()
        self.clip_start_var=tk.StringVar()
        self.clip_end_var=tk.StringVar()
        self.info=tk.StringVar(value='Select a row. Times are seconds in the original WAV.')
        self.filter=tk.StringVar(value='All')
        top=ttk.Frame(self,padding=10);top.pack(fill='x')
        ttk.Label(top,text='Reviewer name:').pack(side='left')
        ttk.Entry(top,textvariable=self.reviewer,width=20).pack(side='left',padx=5)
        ttk.Label(top,text='Show:').pack(side='left',padx=(12,3))
        filters=ttk.Combobox(top,textvariable=self.filter,state='readonly',width=20,
                            values=['All','Pending segments','Approved segments','Excluded segments','Unresolved spans'])
        filters.pack(side='left');filters.bind('<<ComboboxSelected>>',lambda e:self.refresh())
        ttk.Button(top,text='Save approved WAVs and playlist',command=self.save).pack(side='right')
        ttk.Button(top,text='Play approved in order',command=self.play_approved_sequence).pack(side='right',padx=8)
        ttk.Button(top,text='Resolve approved overlaps',command=self.resolve_overlaps).pack(side='right',padx=8)
        ttk.Label(self,text='Approve clips only after listening to their boundaries. Original files stay unchanged.',
                  padding=(10,0)).pack(fill='x')
        table_frame=ttk.Frame(self,padding=10);table_frame.pack(fill='both',expand=True)
        columns=('type','start','end','length','label','status','method')
        self.table=ttk.Treeview(table_frame,columns=columns,show='headings',selectmode='extended')
        for col,caption,width in [('type','Type',95),('start','Source start (s)',110),('end','Source end (s)',110),
                                   ('length','Clip length (s)',105),('label','Label',90),('status','Status',95),('method','Source / reason',400)]:
            self.table.heading(col,text=caption);self.table.column(col,width=width,stretch=col=='method')
        scroll=ttk.Scrollbar(table_frame,orient='vertical',command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.pack(side='left',fill='both',expand=True);scroll.pack(side='right',fill='y')
        self.table.bind('<<TreeviewSelect>>',self.on_select)
        ttk.Label(self,text='Select clips with Ctrl-click or Shift-click, then use Approve selected clips. Apply individual edits first.').pack(anchor='w',padx=10)
        form=ttk.LabelFrame(self,text='Selected item',padding=8);form.pack(fill='x',padx=10,pady=4)
        line=ttk.Frame(form);line.pack(fill='x')
        ttk.Label(line,text='Label:').pack(side='left')
        ttk.Combobox(line,textvariable=self.label,values=list(review.LABELS),state='readonly',width=10).pack(side='left',padx=5)
        ttk.Label(line,text='Source start (s):').pack(side='left')
        ttk.Entry(line,textvariable=self.start_var,width=12).pack(side='left',padx=4)
        ttk.Label(line,text='Source end (s):').pack(side='left')
        ttk.Entry(line,textvariable=self.end_var,width=12).pack(side='left',padx=4)
        ttk.Button(line,text='Apply label / boundaries',command=self.apply).pack(side='left',padx=5)
        ttk.Button(line,text='Approve selected clip',command=self.approve).pack(side='left',padx=5)
        ttk.Button(line,text='Approve selected clips',command=self.approve_selected).pack(side='left',padx=5)
        ttk.Button(line,text='Exclude clip',command=self.exclude).pack(side='left',padx=5)
        line2=ttk.Frame(form);line2.pack(fill='x',pady=(6,0))
        ttk.Button(line2,text='Play full selected',command=self.play_clip).pack(side='left')
        ttk.Button(line2,text='Play source range',command=self.play_source_range).pack(side='left',padx=5)
        ttk.Button(line2,text='Play near start',command=lambda:self.play_edge('start')).pack(side='left',padx=5)
        ttk.Button(line2,text='Play near end',command=lambda:self.play_edge('end')).pack(side='left')
        ttk.Button(line2,text='Stop',command=self.stop_playback).pack(side='left',padx=5)
        ttk.Button(line2,text='Create clip (5–15 s)',command=self.add_clip).pack(side='left',padx=(20,5))
        ttk.Button(line2,text='Mark unresolved reviewed',command=self.mark_reviewed).pack(side='left')
        trim=ttk.Frame(form);trim.pack(fill='x',pady=(6,0))
        ttk.Label(trim,text='Trim generated clip — time within clip (s):').pack(side='left')
        ttk.Label(trim,text='Start:').pack(side='left',padx=(8,0))
        self.clip_start_entry=ttk.Entry(trim,textvariable=self.clip_start_var,width=9);self.clip_start_entry.pack(side='left',padx=3)
        ttk.Label(trim,text='End:').pack(side='left')
        self.clip_end_entry=ttk.Entry(trim,textvariable=self.clip_end_var,width=9);self.clip_end_entry.pack(side='left',padx=3)
        self.preview_trim_button=ttk.Button(trim,text='Preview clip trim',command=self.preview_clip_trim);self.preview_trim_button.pack(side='left',padx=5)
        self.apply_trim_button=ttk.Button(trim,text='Apply clip trim',command=self.apply_clip_trim);self.apply_trim_button.pack(side='left',padx=5)
        ttk.Label(trim,text='5–15 s required. Does not add removed music back into joined clips.').pack(side='left',padx=5)
        automated=ttk.Frame(form);automated.pack(fill='x',pady=(6,0))
        ttk.Button(automated,text='Auto-segment selected unresolved span',command=self.start_resegment).pack(side='left')
        ttk.Label(automated,text='Apply speech/mixed label first. Generated clips remain pending review.').pack(side='left',padx=8)
        ttk.Label(form,textvariable=self.info,wraplength=1200).pack(fill='x',pady=(8,0))
        ttk.Label(self,text='For unresolved spans, set a 5–15 s source range and add a clip. Labeling alone does not resolve or export it.',
                  padding=(10,4)).pack(fill='x')
        self.refresh()
        self.after(100, self.poll_playback_errors)
        self.after(100, self.poll_resegment)

    def refresh(self, select=None):
        self.table.delete(*self.table.get_children())
        for index,item in enumerate(self.data['segments']):
            if self.filter.get()=='Unresolved spans' or self.filter.get()=='Pending segments' and item['status']!='pending' or self.filter.get()=='Approved segments' and item['status']!='approved' or self.filter.get()=='Excluded segments' and item['status']!='excluded':
                continue
            original=item.get('original') or {}
            method=('Edited continuous source' if item['source_mode']=='continuous_source' else ('Trimmed candidate' if item['source_mode']=='trimmed_candidate' else original.get('kind','candidate')))
            length = item['source_end']-item['source_start'] if item['source_mode']=='continuous_source' else (review.candidate_trim_bounds(item)[1]-review.candidate_trim_bounds(item)[0])
            self.table.insert('', 'end', iid=f's{index}', values=('Clip',f"{item['source_start']:.3f}",f"{item['source_end']:.3f}",f'{length:.3f}',item['label'],item['status'],method))
        if self.filter.get() in ('All','Unresolved spans'):
            for index,item in sorted(enumerate(self.data['unresolved']), key=lambda pair: review.unresolved_bounds(pair[1])[0]):
                origin=item['original']
                start,end=review.unresolved_bounds(item)
                self.table.insert('', 'end',iid=f'u{index}',values=('Gap' if item.get('category')=='uncovered_gap' else 'Unresolved',f"{start:.3f}",f"{end:.3f}",f"{end-start:.3f}",item['label'],item['status'],origin.get('reason','')))
        if select and self.table.exists(select):
            self.table.selection_set(select);self.table.see(select)

    def selected(self):
        selected=self.table.selection()
        if not selected:raise ValueError('Select a clip or unresolved span first.')
        if len(selected)!=1:raise ValueError('Select exactly one row for this action.')
        name=selected[0];return name[0],int(name[1:])

    def on_select(self,event=None):
        selected=self.table.selection()
        if len(selected)>1:
            self.info.set(f'{len(selected)} rows selected. Use Approve selected clips; select one row to edit or play it.')
            return
        try: kind,index=self.selected()
        except ValueError:return
        self._form_selected_iid=selected[0]
        if kind=='s':
            item=self.data['segments'][index];self.label.set(item['label'])
            self.start_var.set(str(item['source_start']));self.end_var.set(str(item['source_end']))
            original=item.get('original') or {}; spans=original.get('source_spans',[])
            notes=' '.join(original.get('notes',[]))
            mapping=(' This clip contains multiple source pieces; its displayed bounds are the outer envelope.' if item['source_mode'] in ('original_candidate','trimmed_candidate') and len(spans)>1 else '')
            if item['source_mode'] in ('original_candidate','trimmed_candidate'):
                a,b=review.candidate_trim_bounds(item)
                self.clip_start_var.set(str(a));self.clip_end_var.set(str(b))
                self.set_trim_controls(True)
            else:
                self.clip_start_var.set('');self.clip_end_var.set('')
                self.set_trim_controls(False)
            self.info.set(f"{item['id']} | {item['status']} | {item['source_mode']}.{mapping} {notes}")
        else:
            item=self.data['unresolved'][index];origin=item['original']
            self.clip_start_var.set('');self.clip_end_var.set('');self.set_trim_controls(False)
            self.label.set(item['label'] if item['label'] in review.LABELS else 'speech')
            start,end=review.unresolved_bounds(item)
            self.start_var.set(str(start));self.end_var.set(str(end))
            self.info.set(f"{item['id']} | {item['status']} | {origin.get('reason','')} | {len(origin.get('candidate_files',[]))} original and {len(item.get('generated_candidate_ids',[]))} newly generated candidate(s). Adding or labeling does not automatically mark this span resolved.")

    def show_error(self,exc):messagebox.showerror('Step 4 review',str(exc),parent=self)

    def set_trim_controls(self,enabled):
        state='normal' if enabled else 'disabled'
        for widget in (self.clip_start_entry,self.clip_end_entry,self.preview_trim_button,self.apply_trim_button):
            widget.config(state=state)

    def apply_clip_trim(self):
        try:
            kind,index=self.selected()
            if kind!='s':raise ValueError('Select a generated candidate clip first.')
            item=self.data['segments'][index]
            if self.label.get()!=item['label'] or abs(float(self.start_var.get())-item['source_start'])>1e-8 or abs(float(self.end_var.get())-item['source_end'])>1e-8:
                raise ValueError('Apply the edited label and source times before applying a clip trim.')
            review.set_candidate_trim(self.data,index,self.clip_start_var.get(),self.clip_end_var.get(),self.reviewer.get())
            self.dirty=True;self.refresh(f's{index}')
        except Exception as exc:self.show_error(exc)

    def preview_clip_trim(self):
        try:
            kind,index=self.selected()
            if kind!='s':raise ValueError('Select a generated candidate clip first.')
            item=self.data['segments'][index]
            if item['source_mode']=='continuous_source':raise ValueError('For this source edit use Play source range.')
            a,b=float(self.clip_start_var.get()),float(self.clip_end_var.get())
            if not 0<=a<b<=float(item['original']['duration']):raise ValueError('Clip times must be inside the generated candidate.')
            self.play_file(review.safe_candidate_path(self.data['candidate_root'],item['candidate_file']),a,b)
        except Exception as exc:self.show_error(exc)


    def apply(self):
        try:
            kind,index=self.selected()
            if kind=='u':
                review.label_unresolved(self.data,index,self.label.get(),self.reviewer.get(),
                                        self.start_var.get(),self.end_var.get())
            else:
                item=self.data['segments'][index]
                if item['source_mode'] in ('original_candidate','trimmed_candidate'):
                    clip_start,clip_end=review.candidate_trim_bounds(item)
                    if abs(float(self.clip_start_var.get())-clip_start)>1e-8 or abs(float(self.clip_end_var.get())-clip_end)>1e-8:
                        raise ValueError('Apply the edited clip trim before applying label or source-time changes.')
                start,end=float(self.start_var.get()),float(self.end_var.get())
                changed=abs(start-item['source_start'])>1e-8 or abs(end-item['source_end'])>1e-8
                if changed and item['source_mode'] in ('original_candidate','trimmed_candidate') and len((item.get('original') or {}).get('source_spans',[]))>1:
                    if not messagebox.askyesno('Replace joined clip', 'This clip contains multiple source pieces. Changing its boundaries will replace it with one continuous interval from the original WAV, which can include removed music or silence. Continue?',parent=self):return
                review.set_segment(self.data,index,self.label.get(),start,end,self.reviewer.get())
            self.dirty=True;self.refresh(f'{kind}{index}')
        except Exception as exc:self.show_error(exc)

    def check_applied_clip_edits(self, item):
        if self.label.get()!=item['label'] or abs(float(self.start_var.get())-item['source_start'])>1e-8 or abs(float(self.end_var.get())-item['source_end'])>1e-8:
            raise ValueError('Apply the edited label and boundaries before approving the clip.')
        if item['source_mode'] in ('original_candidate','trimmed_candidate'):
            a,b=review.candidate_trim_bounds(item)
            if abs(float(self.clip_start_var.get())-a)>1e-8 or abs(float(self.clip_end_var.get())-b)>1e-8:
                raise ValueError('Apply the edited clip trim before approving the clip.')

    def approve_selected(self):
        try:
            rows=self.table.selection()
            if not rows:raise ValueError('Select one or more clips to approve.')
            if any(not row.startswith('s') for row in rows):
                raise ValueError('Select clip rows only. Unresolved spans and gaps cannot be approved as clips.')
            if self._form_selected_iid in rows:
                index=int(self._form_selected_iid[1:])
                self.check_applied_clip_edits(self.data['segments'][index])
            count=review.approve_segments(self.data,[int(row[1:]) for row in rows],self.reviewer.get())
            if count:self.dirty=True
            for row in rows:
                if self.filter.get() in ('Pending segments','Excluded segments'):
                    self.table.delete(row)
                else:
                    self.table.set(row,'status','approved')
            self.info.set(f'Approved {count} clips. Previously approved clips were left unchanged. Save approved WAVs and playlist when ready.')
        except Exception as exc:self.show_error(exc)

    def approve(self):
        try:
            kind,index=self.selected()
            if kind!='s':raise ValueError('Select a clip to approve. Use Mark unresolved reviewed for an unresolved span.')
            item=self.data['segments'][index]
            self.check_applied_clip_edits(item)
            review.approve_segment(self.data,index,self.reviewer.get())
            self.dirty=True;self.refresh(f's{index}')
        except Exception as exc:self.show_error(exc)

    def exclude(self):
        try:
            kind,index=self.selected()
            if kind!='s':raise ValueError('Select a clip to exclude.')
            review.exclude_segment(self.data,index,self.reviewer.get())
            self.dirty=True;self.refresh(f's{index}')
        except Exception as exc:self.show_error(exc)

    def add_clip(self):
        try:
            kind,index=self.selected()
            if kind!='u':raise ValueError('Select an unresolved span first.')
            start,end=float(self.start_var.get()),float(self.end_var.get())
            selected_start,selected_end=review.unresolved_bounds(self.data['unresolved'][index])
            if start < selected_start-1/review.RATE or end > selected_end+1/review.RATE:
                if not messagebox.askyesno('Include surrounding audio', 'This clip extends outside the selected span and may overlap other clips or include music or silence. Continue?',parent=self):return
            overlap_count=sum(any(min(end,b)>max(start,a)+1/review.RATE for a,b in review.item_source_ranges(item)) for item in self.data['segments'])
            if overlap_count and not messagebox.askyesno('Overlapping candidate',f'This range overlaps {overlap_count} existing candidate(s). Adding it can duplicate speech. You can instead select and edit an existing clip. Add it anyway?',parent=self):return
            idx=review.add_from_unresolved(self.data,index,start,end,self.label.get(),self.reviewer.get())
            self.filter.set('All');self.dirty=True;self.refresh(f's{idx}')
        except Exception as exc:self.show_error(exc)

    def start_resegment(self):
        try:
            if self._reseg_busy:return
            kind,index=self.selected()
            if kind!='u':raise ValueError('Select an unresolved span or gap first.')
            item=self.data['unresolved'][index]
            start,end=review.unresolved_bounds(item)
            if item['label'] not in ('speech','mixed'):
                raise ValueError('Select speech or mixed and apply the label to this span first.')
            if self.label.get()!=item['label'] or abs(float(self.start_var.get())-start)>1e-8 or abs(float(self.end_var.get())-end)>1e-8:
                raise ValueError('Apply the edited label and source boundaries before auto-segmenting.')
            reviewer=self.reviewer.get().strip()
            if not reviewer:raise ValueError('Enter the reviewer name first.')
            self.stop_playback()
            if self._playback_thread and self._playback_thread.is_alive():
                raise RuntimeError('Audio is still stopping; try again in a moment.')
            self._reseg_busy=True
            self._reseg_dialog=tk.Toplevel(self)
            self._reseg_dialog.title('Auto-segmenting')
            self._reseg_dialog.transient(self.winfo_toplevel())
            self._reseg_dialog.protocol('WM_DELETE_WINDOW',lambda:None)
            self._reseg_message=tk.StringVar(value='Preparing selected source range...')
            ttk.Label(self._reseg_dialog,textvariable=self._reseg_message,wraplength=460,padding=14).pack(fill='x')
            bar=ttk.Progressbar(self._reseg_dialog,mode='indeterminate',length=400)
            bar.pack(padx=14,pady=(0,14));bar.start(12)
            self._reseg_dialog.grab_set()
            source=str(self.source);candidate_root=self.data['candidate_root'];identifier=item['id'];label=item['label']
            def worker():
                try:
                    from step4_resegment import resegment_range
                    records=resegment_range(source,candidate_root,start,end,label,
                        lambda text:self._reseg_queue.put(('progress',text)))
                    self._reseg_queue.put(('done',(index,identifier,reviewer,records)))
                except Exception as exc:
                    self._reseg_queue.put(('error',str(exc)))
            threading.Thread(target=worker,daemon=True).start()
        except Exception as exc:
            self._reseg_busy=False
            if self._reseg_dialog is not None and self._reseg_dialog.winfo_exists():
                self._reseg_dialog.destroy()
            self.show_error(exc)

    def poll_resegment(self):
        try:
            while True:
                kind,result=self._reseg_queue.get_nowait()
                if kind=='progress':
                    if self._reseg_dialog is not None and self._reseg_dialog.winfo_exists():
                        self._reseg_message.set(result)
                    continue
                self._reseg_busy=False
                if self._reseg_dialog is not None and self._reseg_dialog.winfo_exists():
                    self._reseg_dialog.destroy()
                self._reseg_dialog=None
                if isinstance(self,tk.Toplevel):self.grab_set()
                if kind=='error':
                    self.show_error(RuntimeError(result))
                    continue
                index,identifier,reviewer,records=result
                try:
                    if self.data['unresolved'][index]['id']!=identifier:
                        raise ValueError('The selected unresolved span changed during processing.')
                    first=review.add_resegmented_candidates(self.data,index,records,reviewer)
                    self.dirty=True
                    self.filter.set('All')
                    self.refresh(f's{first}')
                    count=sum(r['kind']=='low_energy_fallback' for r in records)
                    messagebox.showinfo('Draft clips generated',
                        f"Generated {len(records)} pending clips covering the selected source range.\n"
                        f"Low-energy fallback clips: {count}.\n\n"
                        'Listen to every boundary, approve the clips you accept, then save approved WAVs. '
                        'The unresolved span remains pending until you review it. Existing candidates may overlap.',parent=self)
                except Exception as exc:self.show_error(exc)
        except queue.Empty:
            pass
        if self.winfo_exists():self.after(100,self.poll_resegment)

    def mark_reviewed(self):
        try:
            kind,index=self.selected()
            if kind!='u':raise ValueError('Select an unresolved span first.')
            item=self.data['unresolved'][index]
            if item['label'] not in review.LABELS:raise ValueError('Apply a label to this unresolved span first.')
            if not messagebox.askyesno('Mark span reviewed', 'This records that you reviewed the whole span. It does not create clips or guarantee that every word is included. Continue?',parent=self):return
            review.mark_unresolved_reviewed(self.data,index,self.reviewer.get())
            self.dirty=True;self.refresh(f'u{index}')
        except Exception as exc:self.show_error(exc)

    def stop_playback(self):
        self._playback_stop.set()
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=1.0)

    def poll_playback_errors(self):
        try:
            while True:
                self.show_error(RuntimeError(self._playback_errors.get_nowait()))
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(100,self.poll_playback_errors)

    def play_file(self,path,start,end):
        """Stream from disk, including long unresolved spans, without loading them into RAM."""
        start,end=float(start),float(end)
        if not 0 <= start < end:
            raise ValueError('Enter a valid playback range.')
        self.stop_playback()
        if self._playback_thread and self._playback_thread.is_alive():
            raise RuntimeError('The previous audio is still stopping; try again in a moment.')
        stop_event=threading.Event()
        self._playback_stop=stop_event
        def worker():
            try:
                with sf.SoundFile(path) as audio:
                    rate=audio.samplerate
                    a=max(0,min(len(audio),round(start*rate)))
                    b=max(a,min(len(audio),round(end*rate)))
                    if b<=a:
                        raise ValueError('The selected playback interval is empty.')
                    audio.seek(a)
                    with sd.OutputStream(samplerate=rate,channels=audio.channels,dtype='float32') as stream:
                        remaining=b-a
                        while remaining and not stop_event.is_set():
                            samples=audio.read(min(8192,remaining),dtype='float32',always_2d=True)
                            if not len(samples):
                                raise IOError('Audio ended before the selected range.')
                            stream.write(samples)
                            remaining-=len(samples)
            except Exception as exc:
                if not stop_event.is_set():
                    self._playback_errors.put(str(exc))
        self._playback_thread=threading.Thread(target=worker,daemon=True)
        self._playback_thread.start()

    def play_approved_sequence(self):
        """Play separate approved clips sequentially without making a merged WAV."""
        try:
            entries=review.approved_playback_queue(self.data,self.source)
            if not entries:
                raise ValueError('Approve at least one reviewed clip first.')
            self.stop_playback()
            if self._playback_thread and self._playback_thread.is_alive():
                raise RuntimeError('The previous audio is still stopping; try again in a moment.')
            stop_event=threading.Event()
            self._playback_stop=stop_event
            self.info.set(f'Playing {len(entries)} approved clips in source order. Stop ends the sequence.')
            def worker():
                try:
                    with sd.OutputStream(samplerate=review.RATE,channels=1,dtype='float32') as stream:
                        for path,start,end,label,identifier in entries:
                            if stop_event.is_set():break
                            with sf.SoundFile(path) as audio:
                                if audio.samplerate!=review.RATE or audio.channels!=1:
                                    raise ValueError(f'Invalid WAV format for {identifier}.')
                                first,last=round(start*audio.samplerate),round(end*audio.samplerate)
                                audio.seek(first);remaining=last-first
                                while remaining and not stop_event.is_set():
                                    samples=audio.read(min(8192,remaining),dtype='float32',always_2d=True)
                                    if not len(samples):raise IOError(f'Audio ended early for {identifier}.')
                                    stream.write(samples);remaining-=len(samples)
                except Exception as exc:
                    if not stop_event.is_set():self._playback_errors.put(str(exc))
            self._playback_thread=threading.Thread(target=worker,daemon=True)
            self._playback_thread.start()
        except Exception as exc:self.show_error(exc)

    def play_clip(self):
        try:
            kind,index=self.selected()
            if kind=='u':
                start,end=float(self.start_var.get()),float(self.end_var.get())
                if not 0<=start<end<=self.data['source']['duration_seconds']:
                    raise ValueError('Enter a valid source range first.')
                self.play_file(self.source,start,end)
                return
            item=self.data['segments'][index]
            if item['source_mode'] in ('original_candidate','trimmed_candidate'):
                path=review.safe_candidate_path(self.data['candidate_root'],item['candidate_file'])
                a,b=review.candidate_trim_bounds(item)
                self.play_file(path,a,b)
            else:self.play_file(self.source,item['source_start'],item['source_end'])
        except Exception as exc:self.show_error(exc)

    def play_source_range(self):
        try:
            start,end=float(self.start_var.get()),float(self.end_var.get())
            if not 0<=start<end<=self.data['source']['duration_seconds']:
                raise ValueError('Enter a valid source range first.')
            self.play_file(self.source,start,end)
        except Exception as exc:self.show_error(exc)

    def play_edge(self,edge):
        try:
            t=float(self.start_var.get() if edge=='start' else self.end_var.get())
            if not 0<=t<=self.data['source']['duration_seconds']:raise ValueError('Time is outside the original WAV.')
            self.play_file(self.source,max(0,t-2),min(self.data['source']['duration_seconds'],t+2))
        except Exception as exc:self.show_error(exc)

    def resolve_overlaps(self):
        try:
            result=review.resolve_approved_overlaps(self.data,self.reviewer.get())
            if result['trimmed'] or result['excluded']:
                self.dirty=True
                self.refresh()
            warnings='\n'.join(result['warnings'][:12])
            if len(result['warnings'])>12:
                warnings+=f"\n... and {len(result['warnings'])-12} more."
            messagebox.showinfo('Overlap cleanup',
                f"Trimmed: {result['trimmed']} clips\nExcluded as fully duplicated: {result['excluded']} clips\n"
                f"Overlapping approved pairs remaining: {len(result['remaining'])}\n"
                f"Clips needing attention: {len(result['warnings'])}\n\n"
                'Automatic cuts are not verified word boundaries. Listen to the changed boundaries before transcription.'
                + ('\n\n' + warnings if warnings else ''),parent=self)
        except Exception as exc:self.show_error(exc)

    def save(self):
        directory=filedialog.askdirectory(title='Choose a parent folder for reviewed clips',parent=self)
        if not directory:return
        try:
            overlaps=review.overlapping_approved(self.data)
            if overlaps and not messagebox.askyesno('Overlapping approved clips',f'{len(overlaps)} pair(s) of approved clips overlap in the original audio, so exported speech may be duplicated. Save anyway?',parent=self):return
            result=review.save_and_export(self.data,self.source,directory)
            self.dirty=False
            messagebox.showinfo('Review saved',f"Exported: {result['exported']} approved clips\nPending clips: {result['pending_segments']}\nExcluded clips: {result['excluded_segments']}\nPending unresolved spans/gaps: {result['pending_unresolved']}\nOverlapping approved pairs: {result['overlapping_pairs']}\n\nSaved in:\n{result['output_dir']}\n\nOpen reviewed_playlist.html to play the WAVs in order.\nstep5_manifest.json lists the separate reviewed files.",parent=self)
        except Exception as exc:self.show_error(exc)

    def close(self):
        if self._reseg_busy:
            messagebox.showinfo('Processing','Please wait until re-segmentation finishes.',parent=self)
            return
        if self.dirty and not messagebox.askyesno('Unsaved changes','Close without saving your review changes?',parent=self):return
        self.stop_playback();self.destroy()


class Step4ReviewPanel(_ReviewBehavior, ttk.Frame):
    def __init__(self, parent, source_wav, metadata_path, reviewer=''):
        super().__init__(parent)
        self.setup_review(source_wav,metadata_path,reviewer)


class Step4ReviewWindow(_ReviewBehavior, tk.Toplevel):
    """Backward-compatible standalone review window."""
    def __init__(self, parent, source_wav, metadata_path, reviewer=''):
        super().__init__(parent)
        self.title('Step 4: Review segments');self.geometry('1300x800')
        self.setup_review(source_wav,metadata_path,reviewer)
        self.protocol('WM_DELETE_WINDOW',self.close)
