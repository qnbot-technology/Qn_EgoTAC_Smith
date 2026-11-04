"""Register the off-centre ball23 coil in an explicitly labelled fixture frame.

The CAD bottom B and the visually labelled root R share an origin, but not
necessarily axes. An electrical rotation about E cannot repair a wrong R/B
anchor: t_R_E must always be R_R_B @ t_B_E.
"""
from __future__ import annotations

import numpy as np


def rigid(matrix):
    t = np.asarray(matrix, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-10)
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-10)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-10)):
        raise ValueError('proper finite SE(3) transform required')
    return t


def derive_fingertip_registration(layout, mount, axis_config, side):
    """Place the CAD coil towards the fingertips, preserving selected E axes.

    The user's anatomical constraint takes precedence over ambiguous face
    numbers. Only the four quarter-turn CAD registrations are searched. No
    camera-space shift, finger fitting, reflection or CAD translation edit is
    involved. The result remains a mounting candidate, not a measurement.
    """
    if not str(layout.get('asset_id', '')).startswith('apriltag_emf_ball') or mount.get('asset_id') != layout['asset_id']:
        raise ValueError('matching Ball geometry required')
    if side not in ('left', 'right') or side not in axis_config['applies_to_sides']:
        raise ValueError('explicit hand registration scope required')
    C0 = rigid(layout['T_render_root_cad_bottom'])
    M0 = rigid(mount['transforms']['T_bottom_emf']['matrix'])
    np.testing.assert_allclose(C0 @ M0, layout['T_render_root_emf_cad_candidate'], atol=1e-12)
    A = rigid(axis_config['T_cad_emf_emf_runtime'])
    if not np.allclose(A[:3, 3], 0, atol=1e-12) or not np.allclose(C0[:3, 3], 0, atol=1e-12):
        raise ValueError('axis selection must not translate E or relocate the fixture bottom')
    previous = C0 @ M0 @ A
    direction = previous[:2, 0]
    if np.linalg.norm(direction) < .99 or np.linalg.norm(M0[:2, 3]) < .001:
        raise ValueError('horizontal fingertip axis and off-centre CAD coil required')
    candidates = []
    for yaw in (0, 90, 180, 270):
        c, s = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
        S = np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])
        C = C0 @ S
        coil = (C @ M0)[:2, 3]
        score = float(coil @ direction / (np.linalg.norm(coil)*np.linalg.norm(direction)))
        candidates.append((score, yaw, C))
    score, yaw, C = max(candidates, key=lambda item: item[0])
    if score < .99:
        raise ValueError('quarter-turn CAD registration cannot satisfy the fingertip constraint')
    M = M0.copy()
    M[:3, :3] = C[:3, :3].T @ previous[:3, :3]
    B = rigid(C @ M)
    np.testing.assert_allclose(B[:3, :3], previous[:3, :3], atol=1e-12)
    np.testing.assert_allclose(M[:3, 3], M0[:3, 3], atol=1e-12)
    return dict(
        side=side, constraint='fingertip_direction',
        anchor_interpretation='below the sloped face towards fingertips; E runtime +X',
        anchor_direction_root=previous[:3, 0].tolist(),
        anchor_tag_ids_used_for_transform=False,
        full_fixture_cad_binding_verified=False, cad_yaw_symmetry_preserved=True,
        T_root_cad_bottom_registered=C.tolist(),
        T_cad_bottom_emf_runtime=M.tolist(), T_root_emf_registered=B.tolist(),
        T_root_emf_previous_direction_only=previous.tolist(),
        relative_cad_anchor_yaw_deg=yaw, coil_anchor_horizontal_alignment_cosine=score,
        yaw_candidates=[dict(relative_yaw_deg=y, alignment_cosine=s) for s,y,_ in candidates],
        delta_origin_root_m=(B[:3, 3]-previous[:3, 3]).tolist(),
        origin_displacement_m=float(np.linalg.norm(B[:3, 3]-previous[:3, 3])),
        translation_source_frame='cad_bottom',
        formula='T_camera_emf = T_camera_root @ T_root_cad_bottom_registered @ T_cad_bottom_emf_runtime',
        fixed_cad_coil_center=True, selected_camera_emf_orientation_unchanged=True,
        camera_tag_observations_unchanged=True, electrical_axes_verified=False, training_ready=False)


def derive_anchor_registration(layout, mount, axis_config, side, anchor_tag_ids):
    """Use one hand's physical tag constraint, retaining selected E orientation.

    Single-face constraints search quarter turns; two-face constraints also
    search intervening 45-degree directions as diagnostics only, without a
    claim of full fixture symmetry. CAD translation remains fixed. This is
    a source-bound candidate, not an electrical calibration.
    """
    if not str(layout.get('asset_id', '')).startswith('apriltag_emf_ball') or mount.get('asset_id') != layout['asset_id']:
        raise ValueError('matching Ball geometry required')
    if side not in ('left', 'right') or side not in axis_config['applies_to_sides']:
        raise ValueError('explicit hand registration scope required')
    ids = list(anchor_tag_ids)
    if len(ids) not in (1, 2) or len(set(ids)) != len(ids):
        raise ValueError('one physical tag or two distinct bounding tags required')
    local_ids = [i - (13 if side == 'right' else 0) for i in ids]
    if any(str(i) not in layout['T_bottom_tag'] or i == 0 for i in local_ids):
        raise ValueError('observed non-top tag binding required for this hand')
    C0 = rigid(layout['T_render_root_cad_bottom'])
    M0 = rigid(mount['transforms']['T_bottom_emf']['matrix'])
    np.testing.assert_allclose(C0 @ M0, layout['T_render_root_emf_cad_candidate'], atol=1e-12)
    A = rigid(axis_config['T_cad_emf_emf_runtime'])
    if not np.allclose(A[:3, 3], 0, atol=1e-12):
        raise ValueError('selected visual axes must not carry a translation')
    if not np.allclose(C0[:3, 3], 0, atol=1e-12):
        raise ValueError('fixture root must be the CAD bottom origin')
    directions = [rigid(layout['T_bottom_tag'][str(i)])[:2, 3] for i in local_ids]
    if any(np.linalg.norm(d) < .001 for d in directions) or np.linalg.norm(M0[:2, 3]) < .001:
        raise ValueError('off-centre coil and radial tag anchor required')
    anchor = np.sum([d/np.linalg.norm(d) for d in directions], axis=0)
    if np.linalg.norm(anchor) < .1:
        raise ValueError('opposed faces do not define an intermediate side')
    candidates = []
    for yaw in range(0, 360, 45 if len(ids) == 2 else 90):
        c, s = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
        S = np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])
        C = C0 @ S
        coil = (C @ M0)[:2, 3]
        score = float(coil @ anchor / (np.linalg.norm(coil)*np.linalg.norm(anchor)))
        candidates.append((score, yaw, C))
    score, yaw, C = max(candidates, key=lambda item: item[0])
    if score < .99:
        raise ValueError('no fixture yaw symmetry places the coil below the selected tag')
    expected_rotation = (C0 @ M0 @ A)[:3, :3]
    M = M0.copy()
    # Express the already selected electrical axes in the newly registered CAD
    # bottom. This is NOT C @ M0 @ A (which would rotate the skeleton again).
    M[:3, :3] = C[:3, :3].T @ expected_rotation
    B = rigid(C @ M)
    np.testing.assert_allclose(B[:3, :3], expected_rotation, atol=1e-12)
    np.testing.assert_allclose(M[:3, 3], M0[:3, 3], atol=1e-12)
    return dict(
        side=side, constraint='tag_face_direction', anchor_tag_ids=ids, local_layout_tag_ids=local_ids,
        anchor_interpretation='below one face' if len(ids) == 1 else 'angular midpoint diagnostic, not a measured mounting',
        azimuth_midpoint_assumption=len(ids) == 2,
        full_fixture_cad_binding_verified=False,
        cad_yaw_symmetry_preserved=bool(yaw % 90 == 0),
        T_root_cad_bottom_registered=C.tolist(),
        T_cad_bottom_emf_runtime=M.tolist(),
        T_root_emf_registered=B.tolist(),
        T_root_emf_previous_direction_only=(C0 @ M0 @ A).tolist(),
        relative_cad_anchor_yaw_deg=yaw,
        coil_anchor_horizontal_alignment_cosine=score,
        yaw_candidates=[dict(relative_yaw_deg=y, alignment_cosine=s) for s,y,_ in candidates],
        delta_origin_root_m=(B[:3, 3]-(C0 @ M0 @ A)[:3, 3]).tolist(),
        origin_displacement_m=float(np.linalg.norm(B[:3, 3]-(C0 @ M0 @ A)[:3, 3])),
        translation_source_frame='cad_bottom',
        axis_rule=dict(x='fingertips; selected visual orientation retained', y='left / thumb-side per user', z='up'),
        formula='T_camera_emf = T_camera_root @ T_root_cad_bottom_registered @ T_cad_bottom_emf_runtime',
        fixed_cad_coil_center=True, selected_camera_emf_orientation_unchanged=True,
        camera_tag_observations_unchanged=True, electrical_axes_verified=False, training_ready=False)


def registered_transforms(config, layout, mount):
    if config.get('schema') not in ('qn_smith.ball23.per_hand_anchor_registration.v1',
                                    'qn_smith.ball26.per_hand_anchor_registration.v1'):
        raise ValueError('explicit per-hand anchor registration required')
    if set(config['registrations']) != {'left', 'right'}:
        raise ValueError('separate left and right registrations required')
    matrices = {}
    for side, saved in config['registrations'].items():
        if saved['side'] != side:
            raise ValueError('hand identity mismatch')
        if saved['constraint'] == 'fingertip_direction':
            result = derive_fingertip_registration(layout, mount, config['selected_axis_config'], side)
        elif saved['constraint'] == 'tag_face_direction':
            result = derive_anchor_registration(layout, mount, config['selected_axis_config'], side, saved['anchor_tag_ids'])
        else:
            raise ValueError('explicit anatomical or face constraint required')
        for key in ('T_root_cad_bottom_registered', 'T_cad_bottom_emf_runtime', 'T_root_emf_registered'):
            np.testing.assert_allclose(saved[key], result[key], atol=1e-12)
        if saved['translation_source_frame'] != 'cad_bottom':
            raise ValueError('CAD offset was relabelled as a different coordinate frame')
        matrices[side] = rigid(result['T_root_emf_registered'])
    return matrices
