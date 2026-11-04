"""Clean tactile and camera-relative wrist-motion display; no label modification."""
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .tactile_hand_panel import hand_image, count_colour, text_at, SIDE_COLOURS

SIZE = (640,1080)
SIDES = ('left','right')


def connected_segments(points, times, valid, last, max_gap_s=.1, max_step_m=.065):
    """Past-only adjacent segments; never join NaNs, dropouts or large jumps."""
    p=np.asarray(points,float);t=np.asarray(times,float);v=np.asarray(valid,bool)
    if p.shape!=(len(t),3) or v.shape!=t.shape or not 0<=last<len(t):
        raise ValueError('Trajectory shapes/index do not match')
    ends=np.arange(1,last+1)
    dt=np.diff(t[:last+1]);step=np.linalg.norm(np.diff(p[:last+1],axis=0),axis=1)
    keep=(v[ends]&v[ends-1]&np.isfinite(p[ends]).all(1)&np.isfinite(p[ends-1]).all(1)
          &(dt>0)&(dt<=max_gap_s)&(step<=max_step_m))
    return np.c_[ends[keep]-1,ends[keep]]


class FixedTrajectoryView:
    """One fixed orthographic view and equal spatial scale for both variants.

    Input axes are original color optical X-right, Y-down, Z-forward (metres).
    Display basis is orthonormal; no independent axis scaling or camera follow.
    Bounds may use the full clip offline, but no future trajectory is drawn.
    """
    def __init__(self, all_points, *, viewport=(84.,663.5,556.,988.5)):
        p=np.asarray(all_points,float).reshape(-1,3)
        p=p[np.isfinite(p).all(1)]
        self.has_finite_positions = bool(len(p))
        self.viewport=np.asarray(viewport,float)
        if (self.viewport.shape!=(4,) or not np.isfinite(self.viewport).all()
                or np.any(self.viewport[2:]<=self.viewport[:2])):
            raise ValueError('Finite positive viewport required')
        self.low=np.floor((p.min(0)-.04)/.1)*.1 if len(p) else np.array([-.2,-.2,0.])
        self.high=np.ceil((p.max(0)+.04)/.1)*.1 if len(p) else np.array([.2,.2,.4])
        a,e=np.deg2rad([24,24])
        # Display world=(camera X,camera Z,-camera Y), screen up is negative v.
        right=np.array([np.cos(a),np.sin(a),0.])
        up=np.array([-np.sin(a)*np.sin(e),np.cos(a)*np.sin(e),np.cos(e)])
        convert=np.array([[1,0,0],[0,0,1],[0,-1,0]])
        self.basis=np.array([right,-up])@convert
        corners=np.array([[x,y,z] for x in (self.low[0],self.high[0])
                          for y in (self.low[1],self.high[1]) for z in (self.low[2],self.high[2])])
        q=corners@self.basis.T;lo,hi=q.min(0),q.max(0)
        self.scale=float(np.min((self.viewport[2:]-self.viewport[:2])/(hi-lo)))
        self.offset=(self.viewport[:2]+self.viewport[2:])*.5-(hi+lo)*.5*self.scale

    def project(self,points):
        return np.asarray(points,float)@self.basis.T*self.scale+self.offset

    def metadata(self):
        return dict(coordinate_frame='original_color_optical_camera_relative',unit='m',
            axes='X right, Y down, Z forward',point='HumanV2 wrist node 0',
            low_m=self.low,high_m=self.high,basis=self.basis,pixels_per_m=self.scale,
            offset_px=self.offset,fixed_view=True,shared_between_versions=True,
            world_camera_motion_removed=False,future_points_drawn=False,
            bounds_from_full_clip_offline=self.has_finite_positions,
            empty_display_grid=not self.has_finite_positions,max_connected_gap_s=.1,max_connected_step_m=.065)

    def background(self,style):
        panel=static_layout(style).copy()
        def line(a,b,col=(58,45,34),width=1):
            q=np.rint(self.project([a,b])).astype(int)
            cv2.line(panel,tuple(q[0]),tuple(q[1]),col,width,cv2.LINE_AA)
        # Ten-centimetre metric floor grid; corner edges give the depth cue.
        for x in np.arange(self.low[0],self.high[0]+.001,.1):
            line([x,self.high[1],self.low[2]],[x,self.high[1],self.high[2]])
        for z in np.arange(self.low[2],self.high[2]+.001,.1):
            line([self.low[0],self.high[1],z],[self.high[0],self.high[1],z])
        origin=self.low.copy()
        for axis,name in enumerate(('X','Y','Z')):
            end=origin.copy();end[axis]=self.high[axis]
            line(origin,end,(113,90,69))
            q=np.rint(self.project(end)).astype(int)
            cv2.putText(panel,name,(int(q[0])+8,int(q[1])+3),cv2.FONT_HERSHEY_SIMPLEX,
                        .45,(147,127,107),1,cv2.LINE_AA)
        return panel


@lru_cache(maxsize=8)
def static_layout(style):
    image=Image.new('RGB',SIZE,(11,18,27));d=ImageDraw.Draw(image)
    for box in ((20,20,620,502),(20,522,620,1060)):
        d.rounded_rectangle(box,radius=24,fill=(18,28,40),outline=(36,49,64),width=1)
    text_at(d,(42,34),'触觉',27,(233,241,249))
    text_at(d,(421,45),'弱',14,(132,153,171))
    text_at(d,(587,45),'强',14,(132,153,171))
    for i,(side,name) in enumerate(zip(SIDES,('左手','右手'))):
        x=42+i*306
        d.ellipse((x,96,x+8,104),fill=tuple(reversed(SIDE_COLOURS[i])))
        text_at(d,(x+19,83),name,21,(198,214,228))
    text_at(d,(42,540),'双手轨迹',27,(233,241,249))
    text_at(d,(492,551),'相机坐标',16,(132,153,171))
    for i,name in enumerate(('左手','右手')):
        x=44+i*112
        d.line((x,600,x+24,600),fill=tuple(reversed(SIDE_COLOURS[i])),width=3)
        text_at(d,(x+33,585),name,17,(176,197,215))
    text_at(d,(43,1015),'网格 10 cm',15,(111,134,153))
    panel=np.asarray(image)[:,:,::-1].copy()
    for x in range(447,575):
        cv2.line(panel,(x,51),(x,58),count_colour((x-447)/127*255,style),1)
    return panel


class MotionDashboard:
    def __init__(self,tactile,points,times,valid,view,style):
        self.tactile=tactile;self.points=points;self.times=np.asarray(times,float)
        self.valid=valid;self.view=view;self.style=style
        self.template=view.background(style)
        self.segments={s:connected_segments(points[s],times,valid[s],len(times)-1) for s in SIDES}

    def render(self,row):
        panel=self.template.copy()
        for i,side in enumerate(SIDES):
            z=self.tactile;valid=bool(z[side+'_tactile_valid'][row])
            peaks=z[side+'_finger_peak_residual_count'][row]
            palm=z[side+'_palm_peak_residual_count'][row]
            hand=hand_image(side,peaks,palm,valid,style=self.style,background_bgr=(40,28,18))
            hand=cv2.resize(hand,(252,324),interpolation=cv2.INTER_AREA)
            # Match the hand illustration's background to the shared card.
            panel[136:460,42+i*306:294+i*306]=hand
            if not valid:
                image=Image.fromarray(panel[:,:,::-1]);d=ImageDraw.Draw(image)
                text_at(d,(137+i*306,467),'缺测',15,(204,155,121))
                panel=np.asarray(image)[:,:,::-1].copy()
            color=np.asarray(SIDE_COLOURS[i],float)
            q=self.view.project(self.points[side])
            segments=self.segments[side];segments=segments[segments[:,1]<=row]
            # Dim accumulated history, with a brighter last 3 seconds.
            for a,b in segments:
                age=self.times[row]-self.times[b]
                weight=.30 if age>3 else .45+.55*(1-age/3)
                ink=tuple(np.rint(color*weight+np.array([40,28,18])*(1-weight)).astype(int).tolist())
                pts=np.rint(q[[a,b]]).astype(int)
                cv2.line(panel,tuple(pts[0]),tuple(pts[1]),ink,2,cv2.LINE_AA)
            if self.valid[side][row] and np.isfinite(q[row]).all():
                pos=tuple(np.rint(q[row]).astype(int))
                cv2.circle(panel,pos,10,tuple(SIDE_COLOURS[i]),2,cv2.LINE_AA)
                cv2.circle(panel,pos,4,(230,240,250),-1,cv2.LINE_AA)
        duration=self.times[-1]-self.times[0];elapsed=self.times[row]-self.times[0]
        cv2.line(panel,(43,1042),(596,1042),(56,42,30),2,cv2.LINE_AA)
        if duration>0:
            cv2.line(panel,(43,1042),(43+round(553*elapsed/duration),1042),(165,141,118),2,cv2.LINE_AA)
        return panel
