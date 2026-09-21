// Turning SAM's low-resolution mask logits into something usable: a binary mask at a chosen size,
// cleaned of stray specks, and traced into a simplified polygon.

export interface MaskOutputs {
  pred_masks: { data: ArrayLike<number>; dims: number[] };
  iou_scores: { data: ArrayLike<number> };
}

export interface BestMask {
  logits: ArrayLike<number>;
  offset: number;
  maskW: number;
  maskH: number;
  /** The part of the mask grid covered by the image (the rest is padding). */
  cropW: number;
  cropH: number;
  score: number;
}

/**
 * Every reading SAM offers of the same prompt — typically a part, a bigger part, and the whole
 * object — and where the image sits inside the mask grid. SAM 1 pads the resized image to a
 * square; SAM 2/3 stretch it, so pass `pad` = null.
 */
export function candidateMasks(outputs: MaskOutputs, resized: [number, number], pad: { height: number; width: number } | null): BestMask[] {
  const scores = outputs.iou_scores.data;
  const logits = outputs.pred_masks.data;
  const [, , count, maskH, maskW] = outputs.pred_masks.dims;
  const [resizedH, resizedW] = resized;
  const shared = {
    logits,
    maskW,
    maskH,
    cropW: (maskW * resizedW) / (pad?.width ?? resizedW),
    cropH: (maskH * resizedH) / (pad?.height ?? resizedH),
  };
  return Array.from({ length: count }, (_, i) => ({ ...shared, offset: i * maskH * maskW, score: scores[i] }));
}

/** Pick the reading SAM itself rates highest. */
export function bestMask(outputs: MaskOutputs, resized: [number, number], pad: { height: number; width: number } | null): BestMask {
  const candidates = candidateMasks(outputs, resized, pad);
  return candidates.reduce((best, c) => (c.score > best.score ? c : best));
}

/** Add `addition` to `mask`, in place. */
export function addMask(mask: Uint8Array, addition: Uint8Array) {
  for (let i = 0; i < mask.length; i++) if (addition[i]) mask[i] = 1;
}

/** Take `removal` out of `mask`, in place. */
export function subtractMask(mask: Uint8Array, removal: Uint8Array) {
  for (let i = 0; i < mask.length; i++) if (removal[i]) mask[i] = 0;
}

/**
 * Close any gaps enclosed by the mask, in place. An annotation is a single ring of points with no
 * holes, so a gap in the middle of a mask is not something the outline can show; filling it keeps
 * what we remember of the object the same as what the user is looking at.
 */
export function fillHoles(mask: Uint8Array, width: number, height: number) {
  // Anything empty that can be reached from the edge of the picture is outside the object.
  const outside = new Uint8Array(mask.length);
  const stack: number[] = [];
  const open = (p: number) => {
    if (mask[p] || outside[p]) return;
    outside[p] = 1;
    stack.push(p);
  };
  for (let x = 0; x < width; x++) {
    open(x);
    open((height - 1) * width + x);
  }
  for (let y = 0; y < height; y++) {
    open(y * width);
    open(y * width + width - 1);
  }
  while (stack.length) {
    const p = stack.pop()!;
    const x = p % width;
    const y = (p / width) | 0;
    if (x > 0) open(p - 1);
    if (x < width - 1) open(p + 1);
    if (y > 0) open(p - width);
    if (y < height - 1) open(p + width);
  }
  for (let i = 0; i < mask.length; i++) if (!mask[i] && !outside[i]) mask[i] = 1;
}

/** Bilinearly upsample one low-res mask (logits) to outW×outH and threshold at 0. */
export function upsampleMask(m: BestMask, outW: number, outH: number) {
  const { logits, offset, maskW, maskH, cropW, cropH } = m;
  const out = new Uint8Array(outW * outH);
  const maxU = Math.max(0, Math.ceil(cropW) - 1);
  const maxV = Math.max(0, Math.ceil(cropH) - 1);
  const x0 = new Int32Array(outW);
  const x1 = new Int32Array(outW);
  const wx = new Float32Array(outW);
  for (let x = 0; x < outW; x++) {
    const u = Math.min(Math.max(((x + 0.5) * cropW) / outW - 0.5, 0), maxU);
    x0[x] = Math.floor(u);
    x1[x] = Math.min(x0[x] + 1, maxU);
    wx[x] = u - x0[x];
  }
  for (let y = 0; y < outH; y++) {
    const v = Math.min(Math.max(((y + 0.5) * cropH) / outH - 0.5, 0), maxV);
    const y0 = Math.floor(v);
    const y1 = Math.min(y0 + 1, maxV, maskH - 1);
    const wy = v - y0;
    const r0 = offset + y0 * maskW;
    const r1 = offset + y1 * maskW;
    const row = y * outW;
    for (let x = 0; x < outW; x++) {
      const top = logits[r0 + x0[x]] * (1 - wx[x]) + logits[r0 + x1[x]] * wx[x];
      const bottom = logits[r1 + x0[x]] * (1 - wx[x]) + logits[r1 + x1[x]] * wx[x];
      if (top * (1 - wy) + bottom * wy > 0) out[row + x] = 1;
    }
  }
  return out;
}

/**
 * SAM masks often include a few stray specks away from the object. They are invisible at a glance
 * but stretch the bounding box, and only one region can become a polygon anyway, so keep a single
 * one: the largest region a click landed on, or simply the largest when no click did. Modifies
 * `mask` in place.
 */
export function keepBestRegion(mask: Uint8Array, width: number, height: number, seeds: Array<[number, number]>) {
  const labels = new Int32Array(mask.length).fill(-1);
  const stack = new Int32Array(mask.length);
  const sizes: number[] = [];
  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || labels[start] >= 0) continue;
    const label = sizes.length;
    let size = 0;
    let top = 0;
    stack[top++] = start;
    labels[start] = label;
    while (top > 0) {
      const p = stack[--top];
      size++;
      const x = p % width;
      const y = (p / width) | 0;
      if (x > 0 && mask[p - 1] && labels[p - 1] < 0) labels[stack[top++] = p - 1] = label;
      if (x < width - 1 && mask[p + 1] && labels[p + 1] < 0) labels[stack[top++] = p + 1] = label;
      if (y > 0 && mask[p - width] && labels[p - width] < 0) labels[stack[top++] = p - width] = label;
      if (y < height - 1 && mask[p + width] && labels[p + width] < 0) labels[stack[top++] = p + width] = label;
    }
    sizes.push(size);
  }
  if (sizes.length <= 1) return;

  const clicked = new Set<number>();
  for (const [sx, sy] of seeds) {
    const cx = Math.min(width - 1, Math.max(0, Math.round(sx)));
    const cy = Math.min(height - 1, Math.max(0, Math.round(sy)));
    // A click can land a pixel or two outside the mask edge, so look in a small neighbourhood.
    let found = false;
    for (let r = 0; r <= 3 && !found; r++) {
      for (let dy = -r; dy <= r && !found; dy++) {
        for (let dx = -r; dx <= r && !found; dx++) {
          const x = cx + dx;
          const y = cy + dy;
          if (x < 0 || y < 0 || x >= width || y >= height) continue;
          const label = labels[y * width + x];
          if (label >= 0) {
            clicked.add(label);
            found = true;
          }
        }
      }
    }
  }
  let keep = -1;
  for (const label of clicked.size ? clicked : sizes.keys()) if (keep < 0 || sizes[label] > sizes[keep]) keep = label;
  for (let i = 0; i < mask.length; i++) if (mask[i] && labels[i] !== keep) mask[i] = 0;
}

// Clockwise neighbours on screen (y down), starting west.
const DIRS: Array<[number, number]> = [[-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1]];
const dirIndex = (dx: number, dy: number) => DIRS.findIndex(([x, y]) => x === dx && y === dy);

/**
 * Moore-neighbour tracing of the outer boundary of the first region found in raster order.
 * Returns pixel coordinates in clockwise order; holes inside the region are ignored.
 */
export function traceOuterContour(mask: Uint8Array, width: number, height: number): Array<[number, number]> {
  let start = -1;
  for (let i = 0; i < mask.length; i++) {
    if (mask[i]) {
      start = i;
      break;
    }
  }
  if (start < 0) return [];
  const isSet = (x: number, y: number) => x >= 0 && y >= 0 && x < width && y < height && mask[y * width + x] === 1;
  const sx = start % width;
  const sy = (start / width) | 0;
  const contour: Array<[number, number]> = [[sx, sy]];
  let cx = sx;
  let cy = sy;
  // The start pixel is the first set pixel in raster order, so its west neighbour is background.
  let back = 0;
  for (let step = 0, maxSteps = mask.length * 4; step < maxSteps; step++) {
    let next = -1;
    for (let k = 1; k <= 8; k++) {
      const d = (back + k) % 8;
      if (isSet(cx + DIRS[d][0], cy + DIRS[d][1])) {
        next = d;
        break;
      }
    }
    if (next < 0) break; // A single isolated pixel.
    const nx = cx + DIRS[next][0];
    const ny = cy + DIRS[next][1];
    // Stop once we're about to repeat the first step from the start pixel: the ring is closed.
    if (cx === sx && cy === sy && contour.length > 1 && nx === contour[1][0] && ny === contour[1][1]) break;
    // The new backtrack is the last background neighbour we checked, relative to the new pixel.
    const prev = (next + 7) % 8;
    back = dirIndex(cx + DIRS[prev][0] - nx, cy + DIRS[prev][1] - ny);
    cx = nx;
    cy = ny;
    contour.push([cx, cy]);
  }
  const last = contour[contour.length - 1];
  if (contour.length > 1 && last[0] === sx && last[1] === sy) contour.pop();
  return contour;
}

function distanceToSegment(p: [number, number], a: [number, number], b: [number, number]) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const lengthSq = dx * dx + dy * dy;
  if (lengthSq === 0) return Math.hypot(p[0] - a[0], p[1] - a[1]);
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / lengthSq));
  return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
}

/** Ramer–Douglas–Peucker on an open polyline, iterative so long contours can't overflow the stack. */
function simplifyOpen(points: Array<[number, number]>, epsilon: number) {
  if (points.length < 3) return points;
  const keep = new Uint8Array(points.length);
  keep[0] = 1;
  keep[points.length - 1] = 1;
  const stack: Array<[number, number]> = [[0, points.length - 1]];
  while (stack.length) {
    const [s, e] = stack.pop()!;
    let maxDistance = 0;
    let index = -1;
    for (let i = s + 1; i < e; i++) {
      const d = distanceToSegment(points[i], points[s], points[e]);
      if (d > maxDistance) {
        maxDistance = d;
        index = i;
      }
    }
    if (maxDistance > epsilon) {
      keep[index] = 1;
      stack.push([s, index], [index, e]);
    }
  }
  return points.filter((_, i) => keep[i]);
}

/** Simplify a closed ring, splitting it at the point farthest from the start so both halves are well-posed. */
export function simplifyRing(ring: Array<[number, number]>, epsilon: number): Array<[number, number]> {
  if (ring.length <= 4) return ring;
  let far = 0;
  let farDistance = -1;
  for (let i = 1; i < ring.length; i++) {
    const d = Math.hypot(ring[i][0] - ring[0][0], ring[i][1] - ring[0][1]);
    if (d > farDistance) {
      farDistance = d;
      far = i;
    }
  }
  const first = simplifyOpen(ring.slice(0, far + 1), epsilon);
  const second = simplifyOpen([...ring.slice(far), ring[0]], epsilon);
  return [...first.slice(0, -1), ...second.slice(0, -1)];
}
