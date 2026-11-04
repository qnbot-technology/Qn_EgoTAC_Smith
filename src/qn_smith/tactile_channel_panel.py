"""Protocol-only tactile cell view. Cell order is not finger anatomy.

The PDF confirms finger membership and 6x4/palm15x16 channel grids, but does
not identify fingertip/pad/base or physical mounting orientation. This view
therefore keeps channel row labels and does not paint anatomical subregions.
"""
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image,ImageDraw

from .tactile_protocol import finger_channels,palm_channels
from .tactile_hand_panel import text_at,SIDE_COLOURS,REGION_LABELS,PALETTE,NEUTRAL,hand_geometry,bezier
from .tactile_display import display_level,neutral_weight

SIZE=(1920,1080)

SCHEMATIC_MAPPING={
    'status':'unconfirmed_anatomical_orientation',
    'finger_rows':'r17 at schematic base to r22 at schematic tip; provisional, not measured',
    'finger_columns':'protocol ascending columns across the unmirrored left template; mirrored right template',
    'palm':'r14 at schematic top to r0 at bottom; 16 ascending columns; provisional',
    'anatomical_tip_pad_localization':False,
    'confirmed':'only finger membership, palm membership and protocol channel order',
    'raw_values_modified':False,
}


@lru_cache(maxsize=2)
def schematic_channel_indices(side):
    """Return per-pixel channel IDs, without claiming physical registration."""
    yy,xx=np.mgrid[:360,:280]
    bones=[((185,255),(239,186),33),((168,197),(178,65),29),
           ((132,188),(130,32),31),((98,193),(83,60),29),((68,211),(40,117),24)]
    indices=[]
    for channels,(a,b,width) in zip(finger_channels(side).values(),bones):
        a,b=np.asarray(a,float),np.asarray(b,float);direction=b-a
        length=np.linalg.norm(direction);n=np.array([-direction[1],direction[0]])/length
        t=((xx-a[0])*direction[0]+(yy-a[1])*direction[1])/length**2
        u=((xx-a[0])*n[0]+(yy-a[1])*n[1])/width+.5
        r=np.minimum((np.clip(t,0,1)*6).astype(int),5)
        c=np.minimum((np.clip(u,0,1)*4).astype(int),3)
        idx=np.asarray(channels).reshape(6,4)[r,c]
        indices.append(idx[:,::-1].copy() if side=='right' else idx)
    r=np.minimum((np.clip((330-yy)/150,0,1)*15).astype(int),14)
    c=np.minimum((np.clip((xx-58)/145,0,1)*16).astype(int),15)
    # Right protocol palm starts at column 0, left starts at column 4.
    palm=np.asarray(palm_channels(side)).reshape(15,16)[r,c]
    if side=='right':palm=palm[:,::-1].copy()
    return indices,palm


def channel_hand_image(side,colors,valid,*,background_bgr=(40,28,18)):
    """Same approved hand outline, channel-wise color; schematic placement only."""
    alpha,masks,palm_mask,shade,shadow,creases,contours=hand_geometry(side)
    indices,palm_indices=schematic_channel_indices(side)
    base=np.full((360,280,3),background_bgr,np.float32)*(1-shadow)
    surface=np.full((360,280,3),NEUTRAL,np.float32)
    if valid:
        surface=surface*(1-palm_mask)+colors[palm_indices]*palm_mask
        for idx,mask in zip(indices,masks):
            surface=surface*(1-mask)+colors[idx]*mask
    else:surface[:]=(85,77,69)
    image=np.clip(base*(1-alpha)+surface*shade*alpha,0,255).astype(np.uint8)
    detail=image.copy()
    for a,b in creases:
        cv2.line(detail,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)),(62,62,58),1,cv2.LINE_AA)
    curves=[((78,226),(105,242),(134,247),(164,241)),
            ((91,206),(117,218),(141,219),(164,214)),
            ((174,233),(153,251),(151,282),(157,299))]
    for curve in curves:
        pts=bezier(*curve)
        if side=='right':pts[:,0]=279-pts[:,0]
        cv2.polylines(detail,[np.rint(pts).astype(np.int32)],False,(59,62,59),1,cv2.LINE_AA)
    image=cv2.addWeighted(image,.70,detail,.30,0)
    cv2.drawContours(image,contours,-1,(155,150,131) if valid else (99,95,86),1,cv2.LINE_AA)
    return image


def channel_colours(values,style,deadband=None):
    """Fixed display transform; optional onset measured above the deadband.

    With a channel deadband, smoothly fade only the displayed RGB over the
    next 12 counts. This avoids the jump caused by a hard residual gate at
    e.g. 8 counts. Source positive residuals and deadbands are never changed.
    """
    values=np.asarray(values,float);missing=~np.isfinite(values)
    safe=np.where(missing,0,values)
    t=display_level(safe,style)*(len(PALETTE)-1)
    a=np.minimum(t.astype(int),len(PALETTE)-2);f=(t-a)[...,None]
    target=PALETTE[a]*(1-f)+PALETTE[a+1]*f
    target=np.where((safe==0)[...,None],np.asarray(NEUTRAL),target)
    if deadband is None:
        w=neutral_weight(safe,style)[...,None]
    else:
        deadband=np.asarray(deadband,float)
        if not np.isfinite(deadband).all() or np.any(deadband<0):
            raise ValueError('Finite nonnegative channel deadbands required')
        margin=np.maximum(safe-deadband,0)
        w=neutral_weight(margin,style)[...,None]
    out=np.rint(np.asarray(NEUTRAL)*(1-w)+target*w).astype(np.uint8)
    out[missing]=(85,77,69)
    return out


@lru_cache(maxsize=1)
def channel_template():
    image=Image.new('RGB',SIZE,(11,18,27));d=ImageDraw.Draw(image)
    text_at(d,(44,23),'触觉通道细分预览',32,(233,241,249))
    text_at(d,(44,76),'按协议编号显示；指尖、指腹与指根位置尚待确认',20,(146,167,187))
    cells=[]
    for si,side in enumerate(('left','right')):
        x=24+si*960
        d.rounded_rectangle((x,128,x+912,1032),radius=24,fill=(18,28,40),outline=(36,49,64))
        text_at(d,(x+28,145),'左手' if si==0 else '右手',28,tuple(reversed(SIDE_COLOURS[si])))
        groups=finger_channels(side)
        for fi,(name,indices) in enumerate(groups.items()):
            gx=x+68+fi*164
            text_at(d,(gx+12,203),REGION_LABELS[fi],21,(208,221,234))
            channels=np.asarray(indices).reshape(6,4)[::-1]
            for row in range(6):
                if fi==0:text_at(d,(gx-40,250+row*34),f'r{22-row}',12,(137,157,177))
                for col in range(4):
                    cells.append((side,int(channels[row,col]),(gx+col*34,250+row*34,32,32)))
            text_at(d,(gx+4,465),'6 × 4 通道',15,(132,153,171))
        text_at(d,(x+68,535),'掌心',23,(208,221,234))
        channels=np.asarray(palm_channels(side)).reshape(15,16)[::-1]
        gx=x+68;gy=586
        for row in range(15):
            for col in range(16):
                cells.append((side,int(channels[row,col]),(gx+col*24,gy+row*24,22,22)))
        text_at(d,(x+496,602),'15 × 16 通道',20,(198,214,228))
        text_at(d,(x+496,651),'固定基线残差',18,(132,153,171))
        text_at(d,(x+496,687),'不代表标定力值',18,(132,153,171))
        text_at(d,(x+496,745),'灰色：基线 / 死区内',18,(132,153,171))
        text_at(d,(x+496,783),'空白通道不参与着色',18,(132,153,171))
        text_at(d,(x+496,874),'相同固定色标',18,(132,153,171))
        text_at(d,(x+496,911),'弱',16,(132,153,171))
        text_at(d,(x+811,911),'强',16,(132,153,171))
    return np.asarray(image)[:,:,::-1].copy(),cells


class TactileChannelPanel:
    def __init__(self,tactile,style):
        self.z=tactile;self.template,self.cells=channel_template()
        self.colors={s:channel_colours(tactile[s+'_positive_residual_count'],style,
                                      tactile[s+'_deadband_count'])
                     for s in ('left','right')}
        self.template=self.template.copy()
        bar=channel_colours(np.linspace(0,255,260),style)
        for si in range(2):
            for i,c in enumerate(bar):
                cv2.line(self.template,(574+si*960+i,925),(574+si*960+i,937),tuple(map(int,c)),1)

    def render(self,row):
        panel=self.template.copy()
        for side,channel,(x,y,w,h) in self.cells:
            color=(tuple(map(int,self.colors[side][row,channel]))
                   if self.z[side+'_tactile_valid'][row] else (85,77,69))
            cv2.rectangle(panel,(x,y),(x+w-1,y+h-1),color,-1)
        image=Image.fromarray(panel[:,:,::-1]);d=ImageDraw.Draw(image)
        for si,side in enumerate(('left','right')):
            if not self.z[side+'_tactile_valid'][row]:
                text_at(d,(760+si*960,151),'缺测',22,(204,155,121))
        return np.asarray(image)[:,:,::-1].copy()
