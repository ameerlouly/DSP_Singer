"""Send known byte counts for multiple streams to the progress tracker.

The event helper lets the checks reproduce changing size estimates, resumed downloads
and the last step after transfer.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_download_and_audio_extraction import DownloadProgress

# Feed the tracker one event with the stream ID and byte counts a download would report.
def event(tracker, ident, downloaded, total=None, status='downloading', estimate=None):
    tracker(dict(status=status, info_dict={'format_id':ident}, downloaded_bytes=downloaded,
                 total_bytes=total, total_bytes_estimate=estimate))

values=[];details=[]
tracker=DownloadProgress({'requested_formats':[{'format_id':'v','filesize':9900},{'format_id':'a','filesize':100}]},values.append,details.append)
event(tracker,'a',100,100,'finished')
assert values[-1]==1, values  # A tiny finished audio stream cannot count as half.
event(tracker,'v',4950,9900)
assert values[-1]==50 and details[-1]['downloaded_bytes']==5050
event(tracker,'v',9900,9900,'finished')
assert 100 not in values and details[-1]['state']=='finishing'
assert details[-1]['downloaded_bytes']==details[-1]['total_bytes']==10000
tracker.report(100);assert values[-1]==100 and details[-1]['state']=='complete'

values=[];details=[]
tracker=DownloadProgress({'format_id':'v','filesize_approx':1000},values.append,details.append)
event(tracker,'v',400)
assert values[-1]==40 and details[-1]['total_is_estimate']
event(tracker,'v',400,2000)
assert values[-1]==20 and not details[-1]['total_is_estimate']
event(tracker,'v',500,estimate=4000)
assert details[-1]['total_bytes']==2000  # An estimate cannot replace a known total.

values=[];details=[]
tracker=DownloadProgress({'requested_formats':[{'format_id':'v'},{'format_id':'a','filesize':100}]},values.append,details.append)
event(tracker,'a',50,100)
assert values[-1]==-1 and details[-1]['total_bytes'] is None
assert details[-1]['downloaded_bytes']==50
event(tracker,'v',450,900)
assert values[-1]==50 and details[-1]['total_bytes']==1000

values=[];details=[]
tracker=DownloadProgress({'format_id':'v','tbr':80,'duration':10},values.append,details.append)
event(tracker,'v',50000)
assert values[-1]==50 and details[-1]['total_is_estimate']
# Resumed bytes are already on disk and count immediately, with no timed ramp.
event(tracker,'v',90000,100000)
assert values[-1]==90 and details[-1]['downloaded_bytes']==90000
print('PASS: byte-weighted video/audio progress, actual counts, unknown totals, bitrate estimates, corrected estimates, resume and reserved final completion.')
