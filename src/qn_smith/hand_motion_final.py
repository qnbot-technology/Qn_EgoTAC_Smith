"""16:9 delivery sidebar: original tactile regions and metric live hand poses.

Hand poses use the same camera-space metres, fixed projection and scale as the
wrist trail. No local magnification, mirror, future lines or pose modification.
"""
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .hand_motion_dashboard import SIDES, connected_segments
from .tactile_hand_panel import hand_image, count_colour, text_at, SIDE_COLOURS
from .tactile_channel_panel import channel_hand_image,channel_colours

CANVAS_SIZE=(2560,1440)
PANEL_SIZE=(640,1440)
RGB_RECT=(0,180,1920,1260)
VIEWPORT=(74.,692.,566.,1310.)


@lru_cache(maxsize=8)
def final_layout(style,detail=False):
    image=Image.new('RGB',PANEL_SIZE,(11,18,27));d=ImageDraw.Draw(image)
    for box in ((20,28,620,514),(20,534,620,1412)):
        d.rounded_rectangle(box,radius=24,fill=(18,28,40),outline=(36,49,64),width=1)
    text_at(d,(42,42),'触觉',27,(233,241,249))
    if detail:text_at(d,(139,53),'通道示意',15,(132,153,171))
    text_at(d,(421,53),'弱',14,(132,153,171))
    text_at(d,(587,53),'强',14,(132,153,171))
    for i,name in enumerate(('左手','右手')):
        x=42+i*306
        d.ellipse((x,104,x+8,112),fill=tuple(reversed(SIDE_COLOURS[i])))
        text_at(d,(x+19,91),name,21,(198,214,228))
        x=44+i*112
        d.line((x,622,x+24,622),fill=tuple(reversed(SIDE_COLOURS[i])),width=3)
        text_at(d,(x+33,607),name,17,(176,197,215))
    text_at(d,(42,556),'双手轨迹与姿态',27,(233,241,249))
    text_at(d,(492,568),'相机坐标',16,(132,153,171))
    text_at(d,(43,1360),'网格 10 cm',15,(111,134,153))
    panel=np.asarray(image)[:,:,::-1].copy()
    for x in range(447,575):
        cv2.line(panel,(x,59),(x,66),count_colour((x-447)/127*255,style),1)
    return panel


def draw_pose(panel,points,valid,view,edges,color):
    """Draw only observed current nodes, in the trail's metric projection."""
    if not valid:return
    q=view.project(points);finite=np.isfinite(q).all(1)
    for a,b in edges:
        if finite[a] and finite[b]:
            p=np.rint(q[[a,b]]).astype(int)
            cv2.line(panel,tuple(p[0]),tuple(p[1]),(18,20,24),6,cv2.LINE_AA)
            cv2.line(panel,tuple(p[0]),tuple(p[1]),color,3,cv2.LINE_AA)
    for point in q[finite]:
        xy=tuple(np.rint(point).astype(int))
        cv2.circle(panel,xy,3,(225,236,246),-1,cv2.LINE_AA)


class FinalMotionDashboard:
    def __init__(self,tactile,skeletons,times,valid,view,style,edges,*,detail=False):
        self.tactile=tactile;self.skeletons=skeletons;self.times=np.asarray(times,float)
        self.valid=valid;self.view=view;self.style=style;self.edges=edges
        if len(edges)!=24 or (4,5) in edges:
            raise ValueError('Expected the verified HumanV2 25-node, 24-edge tree')
        self.detail=detail
        self.colors=({s:channel_colours(tactile[s+'_positive_residual_count'],style,
                                      tactile[s+'_deadband_count']) for s in SIDES}
                     if detail else None)
        self.template=final_layout(style,detail).copy()
        self.points={s:skeletons[s][:,0] for s in SIDES}
        self.projected={s:view.project(self.points[s]) for s in SIDES}
        self.segments={s:connected_segments(self.points[s],times,valid[s],len(times)-1) for s in SIDES}
        self._grid()

    def _grid(self):
        v=self.view
        def line(a,b,col=(58,45,34)):
            q=np.rint(v.project([a,b])).astype(int)
            cv2.line(self.template,tuple(q[0]),tuple(q[1]),col,1,cv2.LINE_AA)
        for x in np.arange(v.low[0],v.high[0]+.001,.1):
            line([x,v.high[1],v.low[2]],[x,v.high[1],v.high[2]])
        for z in np.arange(v.low[2],v.high[2]+.001,.1):
            line([v.low[0],v.high[1],z],[v.high[0],v.high[1],z])
        for axis,name in enumerate(('X','Y','Z')):
            end=v.low.copy();end[axis]=v.high[axis];line(v.low,end,(113,90,69))
            q=np.rint(v.project(end)).astype(int)
            cv2.putText(self.template,name,(int(q[0])+8,int(q[1])+3),cv2.FONT_HERSHEY_SIMPLEX,
                        .45,(147,127,107),1,cv2.LINE_AA)

    def render(self,row):
        panel=self.template.copy()
        for i,side in enumerate(SIDES):
            z=self.tactile;valid=bool(z[side+'_tactile_valid'][row])
            if self.detail:hand=channel_hand_image(side,self.colors[side][row],valid)
            else:
                hand=hand_image(side,z[side+'_finger_peak_residual_count'][row],
                    z[side+'_palm_peak_residual_count'][row],valid,style=self.style,background_bgr=(40,28,18))
            panel[144:468,42+i*306:294+i*306]=cv2.resize(hand,(252,324),interpolation=cv2.INTER_AREA)
            if not valid:
                image=Image.fromarray(panel[:,:,::-1]);d=ImageDraw.Draw(image)
                text_at(d,(137+i*306,475),'缺测',15,(204,155,121))
                panel=np.asarray(image)[:,:,::-1].copy()
            color=np.asarray(SIDE_COLOURS[i],float);q=self.projected[side]
            for a,b in self.segments[side]:
                if b>row:break
                age=self.times[row]-self.times[b]
                weight=.25 if age>3 else .4+.6*(1-age/3)
                ink=tuple(np.rint(color*weight+np.array([40,28,18])*(1-weight)).astype(int).tolist())
                p=np.rint(q[[a,b]]).astype(int)
                cv2.line(panel,tuple(p[0]),tuple(p[1]),ink,2,cv2.LINE_AA)
        # Draw poses after both trails so history never covers the live fingers.
        for i,side in enumerate(SIDES):
            draw_pose(panel,self.skeletons[side][row],self.valid[side][row],
                      self.view,self.edges,SIDE_COLOURS[i])
        duration=self.times[-1]-self.times[0];elapsed=self.times[row]-self.times[0]
        cv2.line(panel,(43,1394),(596,1394),(56,42,30),2,cv2.LINE_AA)
        if duration>0:
            cv2.line(panel,(43,1394),(43+round(553*elapsed/duration),1394),(165,141,118),2,cv2.LINE_AA)
        return panel


def compose_final(rgb,panel):
    if rgb.shape!=(1080,1920,3) or panel.shape!=(1440,640,3):
        raise ValueError('Expected original 1920x1080 RGB and 640x1440 panel')
    canvas=np.full((1440,2560,3),(27,18,11),np.uint8)
    canvas[180:1260,:1920]=rgb
    canvas[:,1920:]=panel
    return canvas
