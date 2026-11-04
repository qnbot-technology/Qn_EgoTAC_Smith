"""Reproducible shaded palm diagrams driven by protocol-defined region counts.

The fixed illustration is not a pose reconstruction or a force map. Five whole
fingers and one whole palm change colour; no within-region contact is inferred.
"""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .tactile_display import display_level, neutral_weight

SIDE_COLOURS = ((70, 210, 255), (255, 160, 80))  # BGR, same as skeletons
REGION_LABELS = ('拇指', '食指', '中指', '无名指', '小指', '掌心')
BACKGROUND = (29, 20, 14)
CARD = (43, 31, 23)
NEUTRAL = (135, 120, 102)
PALETTE = np.array([[168, 131, 81], [222, 191, 58], [185, 219, 69],
                    [105, 224, 200], [70, 183, 250], [94, 101, 250]], float)


def colour(level):
    level = float(np.clip(level, 0, 1))
    if level == 0:
        return NEUTRAL
    t = level*(len(PALETTE)-1)
    a = min(int(t), len(PALETTE)-2)
    return tuple(np.rint(PALETTE[a]*(1-(t-a)) + PALETTE[a+1]*(t-a)).astype(int).tolist())


def count_colour(count, style=None):
    if style is None or style.neutral_fade_count == 0:
        return colour(display_level(count, style))
    weight = float(neutral_weight(count, style))
    target = np.asarray(colour(display_level(count, style)), float)
    return tuple(np.rint(np.asarray(NEUTRAL)*(1-weight) + target*weight).astype(int).tolist())


@lru_cache(maxsize=16)
def font(size):
    path = Path(__file__).resolve().parents[2] / 'assets/fonts/NotoSansCJK-Regular.ttc'
    if not path.is_file():
        raise FileNotFoundError('Chinese panel font required: '+str(path))
    return ImageFont.truetype(str(path), size)


def text_at(draw, xy, value, size=20, fill=(220, 231, 242)):
    draw.text(xy, value, font=font(size), fill=fill)


def bezier(a,b,c,d):
    t=np.linspace(0,1,35)[:,None]
    return (1-t)**3*np.array(a)+3*(1-t)**2*t*np.array(b)+3*(1-t)*t*t*np.array(c)+t**3*np.array(d)


@lru_cache(maxsize=2)
def hand_geometry(side):
    """280x360; anatomical left thumb on viewer right, right on viewer left."""
    if side not in ('left','right'):
        raise ValueError('explicit hand side required')
    scale=3
    shape=(360*scale,280*scale)
    palm=np.zeros(shape,np.uint8)
    # Smooth thenar/hypothenar outline; thumb joins the palm without a seam.
    curves=[((59,192),(53,220),(62,255),(77,282)),
            ((77,282),(87,299),(97,311),(98,331)),
            ((98,331),(97,337),(97,344),(96,353)),
            ((96,353),(123,355),(151,355),(175,353)),
            ((175,353),(171,322),(174,304),(186,279)),
            ((186,279),(195,259),(217,237),(224,218)),
            ((224,218),(209,211),(198,205),(189,196)),
            ((189,196),(169,164),(89,169),(59,192))]
    polygon=np.vstack([bezier(*c) for c in curves])
    cv2.fillPoly(palm,[np.rint(polygon*scale).astype(np.int32)],255)
    bones=[((185,255),(239,186),33),((168,197),(178,65),29),
           ((132,188),(130,32),31),((98,193),(83,60),29),
           ((68,211),(40,117),24)]
    masks=[]; creases=[]; fades=[]
    yy,xx=np.mgrid[:360,:280]
    for p0,p1,width in bones:
        mask=np.zeros(shape,np.uint8)
        start=tuple(np.rint(np.array(p0)*scale).astype(int))
        end=tuple(np.rint(np.array(p1)*scale).astype(int))
        cv2.line(mask,start,end,255,width*scale,cv2.LINE_AA)
        cv2.circle(mask,end,round(width*scale/2),255,-1,cv2.LINE_AA)
        cv2.circle(mask,start,round(width*scale/2),255,-1,cv2.LINE_AA)
        mask=cv2.resize(mask,(280,360),interpolation=cv2.INTER_AREA)
        masks.append(mask)
        a,b=np.asarray(p0,float),np.asarray(p1,float); direction=b-a
        t=((xx-a[0])*direction[0]+(yy-a[1])*direction[1])/np.dot(direction,direction)
        fades.append(np.clip((t+.04)/.34,0,1).astype(np.float32))
        n=np.array([-direction[1],direction[0]])/np.linalg.norm(direction)
        for q in (.37,.70):
            c=a+q*direction; creases.append((c-n*width*.27,c+n*width*.27))
    palm=cv2.resize(palm,(280,360),interpolation=cv2.INTER_AREA)
    union=np.maximum.reduce([palm,*masks])
    if side=='right':
        palm=palm[:,::-1].copy();union=union[:,::-1].copy()
        masks=[m[:,::-1].copy() for m in masks]
        fades=[m[:,::-1].copy() for m in fades]
        creases=[(np.array([279-a[0],a[1]]),np.array([279-b[0],b[1]])) for a,b in creases]
    # Soft edge shading and a broad light are visual depth cues, not sensor data.
    distance=cv2.distanceTransform((union>120).astype(np.uint8),cv2.DIST_L2,5)
    shade=(.68+.28*np.clip(distance/13,0,1)+.10*np.exp(-((xx-108)**2/17000+(yy-160)**2/40000)))[...,None]
    alpha=union.astype(np.float32)[...,None]/255
    region_masks=[m.astype(np.float32)[...,None]/255*f[...,None] for m,f in zip(masks,fades)]
    # Palm signal fills palm only, with the uninstrumented wrist kept neutral.
    palm_alpha=palm.astype(np.float32)[...,None]/255*np.clip((330-yy)/32,0,1)[...,None]
    shadow=cv2.GaussianBlur(union,(25,25),8).astype(np.float32)[...,None]/255*.34
    contours,_=cv2.findContours(union,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return alpha,region_masks,palm_alpha,shade,shadow,creases,contours


def hand_image(side, peaks, palm_peak, valid, *, style=None, background_bgr=None):
    alpha,masks,palm_mask,shade,shadow,creases,contours=hand_geometry(side)
    base=np.full((360,280,3),CARD if background_bgr is None else background_bgr,np.float32)*(1-shadow)
    surface=np.full((360,280,3),NEUTRAL,np.float32)
    if valid:
        fill=np.asarray(count_colour(float(palm_peak), style),np.float32)
        surface=surface*(1-palm_mask)+fill*palm_mask
        for j,mask in enumerate(masks):
            fill=np.asarray(count_colour(float(peaks[j]), style),np.float32)
            surface=surface*(1-mask)+fill*mask
    else:
        surface[:]=(85,77,69)
    surface*=shade
    image=np.clip(base*(1-alpha)+surface*alpha,0,255).astype(np.uint8)
    # Delicate skin folds; no artificial sensor spots or anatomical claims.
    detail=image.copy()
    for a,b in creases:
        cv2.line(detail,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)),(62,62,58),1,cv2.LINE_AA)
    palm_lines=[((78,226),(105,242),(134,247),(164,241)),
                ((91,206),(117,218),(141,219),(164,214)),
                ((174,233),(153,251),(151,282),(157,299))]
    for curve in palm_lines:
        pts=bezier(*curve)
        if side=='right': pts[:,0]=279-pts[:,0]
        cv2.polylines(detail,[np.rint(pts).astype(np.int32)],False,(59,62,59),1,cv2.LINE_AA)
    image=cv2.addWeighted(image,.70,detail,.30,0)
    cv2.drawContours(image,contours,-1,(155,150,131) if valid else (99,95,86),1,cv2.LINE_AA)
    return image


@lru_cache(maxsize=16)
def static_panel(signal='raw', style=None):
    if signal not in ('raw','residual'):
        raise ValueError('explicit raw/residual signal required')
    image=Image.new('RGB',(640,1080),tuple(reversed(BACKGROUND)))
    d=ImageDraw.Draw(image)
    text_at(d,(26,15),'TACTILE  /  触觉感知' if style is None else style.name,27 if style else 29)
    text_at(d,(27,61),'双手五指与掌心 · '+('原始触觉实时映射' if signal=='raw' else '固定基线校正残差'),18,(143,164,184))
    d.rounded_rectangle((26,105,614,165),radius=14,fill=(20,33,47),outline=(39,58,77))
    text_at(d,(40,112),'弱',17,(159,178,195));text_at(d,(580,112),'强',17,(159,178,195))
    legend = ('灰色为零计数   /   固定色标，非牛顿力值' if signal=='raw' else
              '灰色为基线 / 噪声死区内   ·   固定0–255差值色标')
    if style is not None:
        legend = f'死区≥{style.min_deadband:g} · γ={style.gamma:g} · 灰色渐入{style.neutral_fade_count:g}计数'
    text_at(d,(40,140),legend,13,(137,157,178))
    for si,(name,eng) in enumerate((('左手','LEFT'),('右手','RIGHT'))):
        x=18+si*310; accent=tuple(reversed(SIDE_COLOURS[si]))
        d.rounded_rectangle((x,184,x+293,922),radius=22,fill=tuple(reversed(CARD)),outline=(44,62,80),width=1)
        d.rounded_rectangle((x+17,205,x+20,249),radius=2,fill=accent)
        text_at(d,(x+33,198),name,25)
        text_at(d,(x+34,234),eng,12,(132,155,178))
        text_at(d,(x+107,623),'掌面示意',15,(141,163,184))
        for j,name in enumerate(REGION_LABELS):
            y=666+j*39
            text_at(d,(x+19,y),name,18,(186,204,220))
            if j<5: d.line((x+18,y+35,x+275,y+35),fill=(37,51,65))
    text_at(d,(26,951),'颜色表示区域强弱，手型固定、不推断接触点',17,(157,177,196))
    text_at(d,(26,981),'缺测单独标记；掌心仅使用协议指定通道' if signal=='raw' else '统计估计基线，未做空载确认；差值不代表力',17,(157,177,196))
    d.line((26,1026,614,1026),fill=(36,53,70))
    text_at(d,(26,1042),'Human V2  /  Tag3 · Tag16',16,(112,138,162))
    text_at(d,(452,1042),'QN SMITH',16,(132,161,187))
    panel=np.asarray(image)[:,:,::-1].copy()
    for x in range(75,565):
        c = colour((x-75)/489) if style is None else count_colour((x-75)/489*255, style)
        cv2.line(panel,(x,120),(x,131),c,1)
    return panel


def render_panel(z,row,signal='raw', *, style=None):
    panel=static_panel(signal,style).copy(); image=Image.fromarray(panel[:,:,::-1]);d=ImageDraw.Draw(image)
    states=[]
    for si,side in enumerate(('left','right')):
        suffix='_residual_count' if signal=='residual' else '_count'
        valid=bool(z[side+'_tactile_valid'][row]);peaks=z[side+'_finger_peak'+suffix][row]
        palm=float(z[side+'_palm_peak'+suffix][row]); counts=np.r_[peaks,palm]
        if valid and (not np.isfinite(counts).all() or (counts<0).any() or (counts>255).any()):
            raise ValueError('valid tactile counts must be finite and in uint8 range')
        states.append((side,valid,peaks,palm));x=18+si*310
        d.rounded_rectangle((x+185,211,x+273,240),radius=12,fill=(30,61,57) if valid else (62,45,35))
        text_at(d,(x+201,212),'有数据' if valid else '缺测',15,(137,224,192) if valid else (245,182,135))
        for j,count in enumerate(counts):
            level=display_level(count,style) if valid else 0
            label=(str(int(count)) if signal=='raw' else f'{count:.0f}') if valid else '—'
            text_at(d,(x+229,661+j*39),label,22,
                    tuple(reversed(count_colour(count,style))) if valid else (126,143,159))
    panel=np.asarray(image)[:,:,::-1].copy()
    for si,(side,valid,peaks,palm) in enumerate(states):
        x=18+si*310
        panel[259:619,x+7:x+287]=hand_image(side,peaks,palm,valid,style=style)
        for j,count in enumerate(np.r_[peaks,palm]):
            level=display_level(count,style) if valid else 0
            for k in range(18):
                bx=x+108+k*6
                active=valid and level>(k/18)
                cv2.rectangle(panel,(bx,676+j*39),(bx+3,685+j*39),count_colour(count,style) if active else (66,51,39),-1)
    return panel
