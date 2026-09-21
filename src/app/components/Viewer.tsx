import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { isTyping } from '../keyboard';
import { loadImageByName, useStore, type LoadedImage, type PendingSegment } from '../store';
import type { Annotation, BoxShape, ClassDef, Shape } from '../project/types';
import type { Region } from '../segment/protocol';
import { detailer, MAX_DETAIL_PIXELS } from '../image/detail';
import { ENCODE_SIZE, regionPicture, segmenter, useModelStatus, type ModelStatus } from '../segment/segmenter';

type Point = [number, number];
interface View {
  scale: number;
  x: number;
  y: number;
}
type Corner = 'nw' | 'ne' | 'sw' | 'se';
type Drag =
  | { kind: 'pan'; startX: number; startY: number; origin: View }
  | { kind: 'box'; start: Point; current: Point }
  | { kind: 'move'; id: string; start: Point; original: Shape; moved: boolean }
  | { kind: 'corner'; id: string; corner: Corner; original: BoxShape }
  | { kind: 'vertex'; id: string; index: number; original: Point[] }
  | { kind: 'segment'; start: Point; current: Point; button: number; exclude: boolean; add: boolean };

const UNLABELLED = '#9ca3af';
const MIN_BOX_SCREEN_PX = 4;
const CLOSE_POLYGON_SCREEN_PX = 10;

const clamp = (v: number, min: number, max: number) => Math.min(max, Math.max(min, v));
const round = (v: number) => Math.round(v * 100) / 100;

function normalizeBox(a: Point, b: Point): BoxShape {
  return {
    type: 'box',
    x: round(Math.min(a[0], b[0])),
    y: round(Math.min(a[1], b[1])),
    width: round(Math.abs(a[0] - b[0])),
    height: round(Math.abs(a[1] - b[1])),
  };
}

const fullRegion = (image: LoadedImage): Region => ({ x: 0, y: 0, width: image.width, height: image.height });
const fullKey = (image: LoadedImage) => `${image.name}#full`;
const contains = (r: Region, x: number, y: number, w = 0, h = 0) => x >= r.x && y >= r.y && x + w <= r.x + r.width && y + h <= r.y + r.height;

/**
 * How far outside the current outline a click still counts as part of the same object, as a
 * fraction of the outline's longest side. Clicks beyond this start a new object instead.
 */
const SAME_OBJECT_MARGIN = 0.25;
/**
 * The same margin can't be a fraction alone: when the outline so far is only a fragment of the
 * object, a click on the rest of it lands well outside that fragment. So allow at least this
 * fraction of the area being segmented, which grows and shrinks with how far the user is zoomed in.
 */
const SAME_OBJECT_FLOOR = 0.12;

/** Segment within a cropped region (sharper masks for small objects) once zoomed in past this. */
const CROP_WHEN_VISIBLE_FRACTION_BELOW = 0.4;
/** Fetch full-resolution detail once the view is magnifying the display copy by more than this. */
const DETAIL_ZOOM_FACTOR = 1.2;
/** Longest side of a detail tile; beyond this the memory cost outweighs what the screen can show. */
const MAX_DETAIL_TILE = 3000;
/** Wait for panning and zooming to settle before fetching detail. */
const DETAIL_DELAY_MS = 250;
/** Always leave at least this much of the image on screen, so it can't be zoomed or panned away. */
const KEEP_ON_SCREEN_PX = 60;

/** Is this click on, or close to, the outline we're working on? */
function nearPolygon([x, y]: Point, polygon: Array<[number, number]>, region: Region) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [px, py] of polygon) {
    if (px < minX) minX = px;
    if (py < minY) minY = py;
    if (px > maxX) maxX = px;
    if (py > maxY) maxY = py;
  }
  const margin = Math.max(
    Math.max(maxX - minX, maxY - minY) * SAME_OBJECT_MARGIN,
    Math.max(region.width, region.height) * SAME_OBJECT_FLOOR,
  );
  return x >= minX - margin && x <= maxX + margin && y >= minY - margin && y <= maxY + margin;
}

/** Keep part of the image within the viewport, whatever the zoom or pan. */
function keepOnScreen(view: View, image: LoadedImage, el: HTMLElement): View {
  const width = image.width * view.scale;
  const height = image.height * view.scale;
  return {
    scale: view.scale,
    x: clamp(view.x, Math.min(KEEP_ON_SCREEN_PX - width, 0), Math.max(el.clientWidth - KEEP_ON_SCREEN_PX, 0)),
    y: clamp(view.y, Math.min(KEEP_ON_SCREEN_PX - height, 0), Math.max(el.clientHeight - KEEP_ON_SCREEN_PX, 0)),
  };
}

function translate(shape: Shape, dx: number, dy: number, w: number, h: number): Shape {
  if (shape.type === 'box') {
    return { ...shape, x: round(clamp(shape.x + dx, 0, w - shape.width)), y: round(clamp(shape.y + dy, 0, h - shape.height)) };
  }
  const xs = shape.points.map((p) => p[0]);
  const ys = shape.points.map((p) => p[1]);
  const cdx = clamp(dx, -Math.min(...xs), w - Math.max(...xs));
  const cdy = clamp(dy, -Math.min(...ys), h - Math.max(...ys));
  return { ...shape, points: shape.points.map(([x, y]) => [round(x + cdx), round(y + cdy)]) };
}

export function Viewer() {
  const image = useStore((s) => s.image);
  const doc = useStore((s) => s.doc);
  const imageError = useStore((s) => s.imageError);
  const tool = useStore((s) => s.tool);
  const selectedId = useStore((s) => s.selectedId);
  const classes = useStore((s) => s.project.classes);
  const activeClassId = useStore((s) => s.activeClassId);
  const pending = useStore((s) => s.pendingSegment);
  const modelStatus = useModelStatus((s) => s.status);

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [view, setView] = useState<View>({ scale: 1, x: 0, y: 0 });
  const viewRef = useRef(view);
  viewRef.current = view;
  const imageRef = useRef(image);
  imageRef.current = image;
  const dragRef = useRef<Drag | null>(null);
  const [drag, setDrag] = useState<Drag | null>(null);
  const [draft, setDraft] = useState<Point[]>([]);
  const [hover, setHover] = useState<Point | null>(null);
  const spaceHeld = useRef(false);

  const updateDrag = (next: Drag | null) => {
    dragRef.current = next;
    setDrag(next);
  };

  /** True until the user zooms or pans, so window resizes keep the image fitted. */
  const fitted = useRef(true);
  const fit = useCallback(() => {
    const el = containerRef.current;
    if (!el || !image || el.clientWidth === 0) return;
    const { clientWidth: cw, clientHeight: ch } = el;
    const scale = Math.min(cw / image.width, ch / image.height) * 0.96;
    fitted.current = true;
    setView({ scale, x: (cw - image.width * scale) / 2, y: (ch - image.height * scale) / 2 });
  }, [image]);

  /** Zoom about the middle of the viewport, for the on-screen buttons. */
  const zoomBy = useCallback((factor: number) => {
    const el = containerRef.current;
    if (!el) return;
    fitted.current = false;
    const mx = el.clientWidth / 2;
    const my = el.clientHeight / 2;
    setView((v) => {
      const scale = clamp(v.scale * factor, 0.01, 40);
      const k = scale / v.scale;
      const next = { scale, x: mx - (mx - v.x) * k, y: my - (my - v.y) * k };
      return imageRef.current ? keepOnScreen(next, imageRef.current, el) : next;
    });
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver(() => {
      if (fitted.current) fit();
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [fit]);

  // Draw the bitmap and fit it to the viewport whenever the image changes.
  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !image) return;
    canvas.width = image.bitmap.width;
    canvas.height = image.bitmap.height;
    canvas.getContext('2d')!.drawImage(image.bitmap, 0, 0);
    fit();
    setDraft([]);
    updateDrag(null);
  }, [image, fit]);

  // Leaving the polygon tool abandons an unfinished polygon.
  useEffect(() => {
    if (tool !== 'polygon') setDraft([]);
  }, [tool]);

  const toWorld = useCallback((clientX: number, clientY: number): Point => {
    const rect = containerRef.current!.getBoundingClientRect();
    const v = viewRef.current;
    return [(clientX - rect.left - v.x) / v.scale, (clientY - rect.top - v.y) / v.scale];
  }, []);

  const clampToImage = (p: Point): Point => (image ? [clamp(p[0], 0, image.width), clamp(p[1], 0, image.height)] : p);

  // Wheel zoom around the cursor. Needs a non-passive listener to stop the page scrolling.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      fitted.current = false;
      const rect = el.getBoundingClientRect();
      const mx = event.clientX - rect.left;
      const my = event.clientY - rect.top;
      setView((v) => {
        const scale = clamp(v.scale * Math.exp(-event.deltaY * 0.0015), 0.01, 40);
        const k = scale / v.scale;
        const next = { scale, x: mx - (mx - v.x) * k, y: my - (my - v.y) * k };
        return imageRef.current ? keepOnScreen(next, imageRef.current, el) : next;
      });
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  const finishPolygon = useCallback(() => {
    // A double-click to finish also adds two clicks' worth of points at the same spot; drop them.
    const minGap = 2 / viewRef.current.scale;
    const points = draft.filter((p, i) => i === 0 || Math.hypot(p[0] - draft[i - 1][0], p[1] - draft[i - 1][1]) > minGap);
    if (points.length >= 3) useStore.getState().addAnnotation({ type: 'polygon', points: points.map(([x, y]) => [round(x), round(y)]) });
    setDraft([]);
  }, [draft]);

  // --- One-click segmentation -----------------------------------------------------------------

  // Load the model the first time the Segment tool is chosen.
  useEffect(() => {
    if (tool === 'segment') segmenter.load().catch(() => undefined);
  }, [tool]);

  // Once the model is ready, encode the current image and the next one in the background,
  // so the first click on each image is answered straight away.
  useEffect(() => {
    if (modelStatus.kind !== 'ready' || !image) return;
    segmenter.cancelBackground();
    const encodeFull = (img: LoadedImage) =>
      segmenter.encode(fullKey(img), fullRegion(img), () => regionPicture(img.bitmap, img.width, fullRegion(img)), 'background').catch(() => undefined);
    void encodeFull(image);
    const { images } = useStore.getState();
    const next = images[images.findIndex((i) => i.name === image.name) + 1];
    if (next) loadImageByName(next.name).then(encodeFull, () => undefined);
  }, [modelStatus.kind, image]);

  /** Crops encoded while zoomed in on this image, reused when the next click falls inside one. */
  const crops = useRef<Array<{ key: string; region: Region }>>([]);
  useEffect(() => {
    crops.current = [];
  }, [image]);

  /** The part of the image currently on screen, in image pixels, optionally grown by a margin. */
  const visibleRegion = useCallback((img: LoadedImage, margin = 0): Region => {
    const el = containerRef.current!;
    const v = viewRef.current;
    const x0 = clamp(-v.x / v.scale, 0, img.width);
    const y0 = clamp(-v.y / v.scale, 0, img.height);
    const x1 = clamp((el.clientWidth - v.x) / v.scale, 0, img.width);
    const y1 = clamp((el.clientHeight - v.y) / v.scale, 0, img.height);
    const mx = (x1 - x0) * margin;
    const my = (y1 - y0) * margin;
    const left = Math.floor(clamp(x0 - mx, 0, img.width));
    const top = Math.floor(clamp(y0 - my, 0, img.height));
    return {
      x: left,
      y: top,
      width: Math.max(1, Math.ceil(clamp(x1 + mx, 0, img.width)) - left),
      height: Math.max(1, Math.ceil(clamp(y1 + my, 0, img.height)) - top),
    };
  }, []);

  /** Pick what to encode for a new object: the whole image, or the zoomed-in view for small objects. */
  function chooseRegion(img: LoadedImage, x: number, y: number, box?: BoxShape): { key: string; region: Region } {
    const visible = visibleRegion(img);
    const visibleArea = visible.width * visible.height;
    const full = { key: fullKey(img), region: fullRegion(img) };
    if (visibleArea >= CROP_WHEN_VISIBLE_FRACTION_BELOW * img.width * img.height || visible.width < 16 || visible.height < 16) return full;

    const fits = (r: Region) => contains(r, x, y) && (!box || contains(r, box.x, box.y, box.width, box.height));
    const reusable = crops.current.find(({ region: r }) => fits(r) && r.width * r.height <= 2.5 * visibleArea);
    if (reusable) return reusable;

    // A margin around the visible area gives the model context at the edges of the view.
    const region = visibleRegion(img, 0.1);
    if (!fits(region)) return full;
    const crop = { key: `${img.name}#${region.x},${region.y},${region.width},${region.height}`, region };
    crops.current = [crop, ...crops.current].slice(0, 6);
    return crop;
  }

  const fileFor = useCallback(async (name: string) => {
    const entry = useStore.getState().images.find((i) => i.name === name);
    if (!entry) throw new Error(`${name} is no longer in this folder`);
    return entry.handle.getFile();
  }, []);

  /** Can we do better than the display copy for this image? */
  const canUseDetail = (img: LoadedImage) => img.bitmap.width < img.width && img.width * img.height <= MAX_DETAIL_PIXELS;

  /** The picture handed to the model: cut from the original file when that is sharper than the display copy. */
  const segmentPicture = useCallback(
    async (img: LoadedImage, region: Region) => {
      const isCrop = region.width < img.width || region.height < img.height;
      if (isCrop && canUseDetail(img)) {
        try {
          return await detailer.crop(img.name, await fileFor(img.name), region, ENCODE_SIZE, true);
        } catch {
          // Fall back to the display copy.
        }
      }
      return regionPicture(img.bitmap, img.width, region);
    },
    [fileFor],
  );

  // --- Full-resolution detail while zoomed in ---------------------------------------------------

  const [tile, setTile] = useState<{ name: string; region: Region; bitmap: ImageBitmap } | null>(null);
  const tileRef = useRef<typeof tile>(null);
  const tileSeq = useRef(0);
  const tileCanvasRef = useRef<HTMLCanvasElement>(null);
  const showTile = useCallback((next: typeof tile) => {
    tileRef.current?.bitmap.close();
    tileRef.current = next;
    setTile(next);
  }, []);

  useEffect(() => {
    if (!image) return;
    tileSeq.current++;
    if (tileRef.current && tileRef.current.name !== image.name) showTile(null);
    const displayScale = image.bitmap.width / image.width;
    // The display copy already has every pixel the screen can show.
    if (!canUseDetail(image) || view.scale <= displayScale * DETAIL_ZOOM_FACTOR) {
      if (tileRef.current) showTile(null);
      return;
    }
    const timer = setTimeout(async () => {
      const seq = ++tileSeq.current;
      const region = visibleRegion(image, 0.15);
      if (region.width < 8 || region.height < 8) return;
      try {
        const maxSize = Math.min(MAX_DETAIL_TILE, Math.ceil(Math.max(region.width, region.height) * view.scale * devicePixelRatio));
        const bitmap = await detailer.crop(image.name, await fileFor(image.name), region, maxSize, false);
        if (seq !== tileSeq.current) bitmap.close();
        else showTile({ name: image.name, region, bitmap });
      } catch {
        // Keep showing the display copy.
      }
    }, DETAIL_DELAY_MS);
    return () => clearTimeout(timer);
  }, [image, view, visibleRegion, fileFor, showTile]);

  useEffect(() => {
    const canvas = tileCanvasRef.current;
    if (!canvas || !tile) return;
    canvas.width = tile.bitmap.width;
    canvas.height = tile.bitmap.height;
    canvas.getContext('2d')!.drawImage(tile.bitmap, 0, 0);
  }, [tile]);

  useEffect(() => () => showTile(null), [showTile]);

  const segmentSeq = useRef(0);
  async function runSegment(next: PendingSegment) {
    const img = image;
    if (!img) return;
    const seq = ++segmentSeq.current;
    store().setPendingSegment({ ...next, busy: true, error: null });
    try {
      const box = next.box && { x: next.box.x, y: next.box.y, width: next.box.width, height: next.box.height };
      const result = await segmenter.segment(next.key, next.region, () => segmentPicture(img, next.region), next.points, box);
      if (seq !== segmentSeq.current || store().pendingSegment?.key !== next.key) return;
      store().setPendingSegment({
        ...next,
        polygon: result.polygon,
        score: result.score,
        busy: false,
        error: result.polygon ? null : 'Nothing found there. Try another spot.',
      });
    } catch (err) {
      if (seq !== segmentSeq.current || store().pendingSegment?.key !== next.key) return;
      store().setPendingSegment({ ...next, busy: false, error: err instanceof Error ? err.message : String(err) });
    }
  }
  const runSegmentRef = useRef(runSegment);
  runSegmentRef.current = runSegment;

  function segmentClick(p: Point, exclude: boolean, add: boolean) {
    if (!image) return;
    const current = store().pendingSegment;
    const point = { x: round(p[0]), y: round(p[1]), positive: !exclude };
    // Clicking a spot that's already a prompt (e.g. a double-click) would just repeat the same request.
    const samePoint = current?.points.some((q) => Math.hypot(q.x - point.x, q.y - point.y) * viewRef.current.scale < 4);
    if (current && samePoint) return;
    // Right-click removes an area from the outline we're working on; Ctrl-click always adds to it.
    // A plain click adds to it too when it lands on or near it, and otherwise starts a new object.
    const refine = current && (exclude || add || (current.polygon !== null && nearPolygon(p, current.polygon, current.region)));
    if (refine) {
      void runSegment({ ...current, points: [...current.points, point] });
      return;
    }
    if (exclude) return;
    store().commitPendingSegment();
    const { key, region } = chooseRegion(image, p[0], p[1]);
    void runSegment({ key, region, points: [point], box: null, polygon: null, busy: true, error: null });
  }

  function segmentBox(box: BoxShape) {
    if (!image) return;
    store().commitPendingSegment();
    const { key, region } = chooseRegion(image, box.x + box.width / 2, box.y + box.height / 2, box);
    void runSegment({ key, region, points: [], box, polygon: null, busy: true, error: null });
  }

  // Keys that belong to the viewer. Registered in the capture phase so they win over global shortcuts.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (isTyping(event)) return;
      if (event.code === 'Space') {
        spaceHeld.current = true;
        event.preventDefault();
        return;
      }
      if (event.key === 'f' || event.key === 'F') {
        fit();
        return;
      }
      const pendingNow = useStore.getState().pendingSegment;
      if (pendingNow && ['Enter', 'Escape', 'Backspace'].includes(event.key)) {
        if (event.key === 'Enter') useStore.getState().commitPendingSegment();
        else if (event.key === 'Escape') useStore.getState().setPendingSegment(null);
        else if (pendingNow.points.length > 1) void runSegmentRef.current({ ...pendingNow, points: pendingNow.points.slice(0, -1) });
        else useStore.getState().setPendingSegment(null);
        event.preventDefault();
        event.stopPropagation();
        return;
      }
      if (draft.length === 0) return;
      if (event.key === 'Enter') finishPolygon();
      else if (event.key === 'Escape') setDraft([]);
      else if (event.key === 'Backspace') setDraft((d) => d.slice(0, -1));
      else return;
      event.preventDefault();
      event.stopPropagation();
    };
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.code === 'Space') spaceHeld.current = false;
    };
    window.addEventListener('keydown', onKeyDown, true);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown, true);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, [draft, finishPolygon, fit]);

  const store = useStore.getState;

  function onPointerDown(event: React.PointerEvent<HTMLDivElement>) {
    if (!image) return;
    const panRequested = event.button === 1 || (event.button === 0 && spaceHeld.current);
    if (panRequested || (event.button === 0 && tool === 'select')) {
      if (!panRequested) store().select(null);
      event.currentTarget.setPointerCapture(event.pointerId);
      updateDrag({ kind: 'pan', startX: event.clientX, startY: event.clientY, origin: viewRef.current });
      return;
    }
    if (tool === 'segment' && (event.button === 0 || event.button === 2)) {
      const p = clampToImage(toWorld(event.clientX, event.clientY));
      event.currentTarget.setPointerCapture(event.pointerId);
      updateDrag({
        kind: 'segment',
        start: p,
        current: p,
        button: event.button,
        exclude: event.button === 2 || event.shiftKey || event.altKey,
        add: event.ctrlKey || event.metaKey,
      });
      return;
    }
    if (event.button !== 0) return;
    const p = clampToImage(toWorld(event.clientX, event.clientY));
    if (tool === 'box') {
      event.currentTarget.setPointerCapture(event.pointerId);
      updateDrag({ kind: 'box', start: p, current: p });
    } else if (tool === 'polygon') {
      const first = draft[0];
      const closeEnough = first && Math.hypot(first[0] - p[0], first[1] - p[1]) * viewRef.current.scale < CLOSE_POLYGON_SCREEN_PX;
      if (draft.length >= 3 && closeEnough) finishPolygon();
      else setDraft((d) => [...d, p]);
    }
  }

  function onPointerMove(event: React.PointerEvent<HTMLDivElement>) {
    if (!image) return;
    const d = dragRef.current;
    const p = toWorld(event.clientX, event.clientY);
    if (tool === 'polygon') setHover(clampToImage(p));
    if (!d) return;
    if (d.kind === 'pan') {
      fitted.current = false;
      const panned = { ...d.origin, x: d.origin.x + event.clientX - d.startX, y: d.origin.y + event.clientY - d.startY };
      setView(keepOnScreen(panned, image, event.currentTarget));
    } else if (d.kind === 'box' || d.kind === 'segment') {
      updateDrag({ ...d, current: clampToImage(p) });
    } else if (d.kind === 'move') {
      if (!d.moved) store().beginEdit();
      store().updateShape(d.id, translate(d.original, p[0] - d.start[0], p[1] - d.start[1], image.width, image.height), { record: false });
      if (!d.moved) updateDrag({ ...d, moved: true });
    } else if (d.kind === 'corner') {
      const { x, y, width, height } = d.original;
      const fixed: Point = [d.corner.includes('w') ? x + width : x, d.corner.includes('n') ? y + height : y];
      store().updateShape(d.id, normalizeBox(fixed, clampToImage(p)), { record: false });
    } else if (d.kind === 'vertex') {
      const points = d.original.map((pt, i) => (i === d.index ? clampToImage(p).map(round) as Point : pt));
      store().updateShape(d.id, { type: 'polygon', points }, { record: false });
    }
  }

  function onPointerUp() {
    const d = dragRef.current;
    if (d?.kind === 'box') {
      const box = normalizeBox(d.start, d.current);
      const minSize = MIN_BOX_SCREEN_PX / viewRef.current.scale;
      if (box.width >= minSize && box.height >= minSize) store().addAnnotation(box);
    } else if (d?.kind === 'segment') {
      const box = normalizeBox(d.start, d.current);
      const dragged = Math.max(box.width, box.height) * viewRef.current.scale > MIN_BOX_SCREEN_PX * 2;
      if (dragged && d.button === 0 && !d.exclude) segmentBox(box);
      else segmentClick(d.start, d.exclude, d.add);
    }
    updateDrag(null);
  }

  function startShapeDrag(event: React.PointerEvent, annotation: Annotation) {
    if (tool !== 'select' || event.button !== 0 || spaceHeld.current) return;
    event.stopPropagation();
    containerRef.current!.setPointerCapture(event.pointerId);
    store().select(annotation.id);
    updateDrag({ kind: 'move', id: annotation.id, start: toWorld(event.clientX, event.clientY), original: annotation.shape, moved: false });
  }

  function startCornerDrag(event: React.PointerEvent, annotation: Annotation, corner: Corner) {
    if (annotation.shape.type !== 'box' || event.button !== 0) return;
    event.stopPropagation();
    containerRef.current!.setPointerCapture(event.pointerId);
    store().beginEdit();
    updateDrag({ kind: 'corner', id: annotation.id, corner, original: annotation.shape });
  }

  function startVertexDrag(event: React.PointerEvent, annotation: Annotation, index: number) {
    if (annotation.shape.type !== 'polygon' || event.button !== 0) return;
    event.stopPropagation();
    if (event.altKey) {
      // Alt-click removes a vertex, keeping at least a triangle.
      if (annotation.shape.points.length > 3) {
        store().updateShape(annotation.id, { type: 'polygon', points: annotation.shape.points.filter((_, i) => i !== index) });
      }
      return;
    }
    containerRef.current!.setPointerCapture(event.pointerId);
    store().beginEdit();
    updateDrag({ kind: 'vertex', id: annotation.id, index, original: annotation.shape.points });
  }

  if (imageError) return <div className="viewer empty">Couldn't open this image: {imageError}</div>;

  const classById = new Map<string, ClassDef>(classes.map((c) => [c.id, c]));
  const s = view.scale;
  const handleR = 5 / s;
  const activeColor = classById.get(activeClassId ?? '')?.color ?? UNLABELLED;
  const cursor = drag?.kind === 'pan' ? 'grabbing' : tool === 'select' ? 'grab' : pending?.busy ? 'progress' : 'crosshair';

  return (
    <div
      ref={containerRef}
      className="viewer"
      style={{ cursor }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onPointerLeave={() => setHover(null)}
      onDoubleClick={() => tool === 'polygon' && finishPolygon()}
      onContextMenu={(e) => e.preventDefault()}
    >
      {!image && <div className="viewer-loading">Loading image…</div>}
      {image && (
        <div className="world" style={{ width: image.width, height: image.height, transform: `matrix(${s},0,0,${s},${view.x},${view.y})` }}>
          <canvas ref={canvasRef} className="world-image" />
          {tile && tile.name === image.name && (
            <canvas
              ref={tileCanvasRef}
              className="world-detail"
              style={{ left: tile.region.x, top: tile.region.y, width: tile.region.width, height: tile.region.height }}
            />
          )}
          <svg className="world-overlay" width={image.width} height={image.height} viewBox={`0 0 ${image.width} ${image.height}`}>
            {doc?.annotations.map((a) => {
              const cls = a.classId ? classById.get(a.classId) : undefined;
              const color = cls?.color ?? UNLABELLED;
              const selected = a.id === selectedId;
              const common = {
                className: `shape${selected ? ' selected' : ''}${cls ? '' : ' unlabelled'}`,
                stroke: color,
                fill: color,
                style: { pointerEvents: tool === 'select' ? ('all' as const) : ('none' as const) },
                onPointerDown: (e: React.PointerEvent) => startShapeDrag(e, a),
              };
              const bounds =
                a.shape.type === 'box'
                  ? a.shape
                  : { x: Math.min(...a.shape.points.map((p) => p[0])), y: Math.min(...a.shape.points.map((p) => p[1])) };
              return (
                <g key={a.id}>
                  {a.shape.type === 'box' ? (
                    <rect {...common} x={a.shape.x} y={a.shape.y} width={a.shape.width} height={a.shape.height} />
                  ) : (
                    <polygon {...common} points={a.shape.points.map((p) => p.join(',')).join(' ')} />
                  )}
                  <text className="shape-label" x={bounds.x} y={bounds.y - 4 / s} fontSize={12 / s} strokeWidth={3 / s} fill={color}>
                    {cls?.name ?? 'Unlabelled'}
                  </text>
                  {selected && tool === 'select' && a.shape.type === 'box' &&
                    (['nw', 'ne', 'sw', 'se'] as Corner[]).map((corner) => {
                      const shape = a.shape as BoxShape;
                      return (
                        <rect
                          key={corner}
                          className={`handle handle-${corner}`}
                          x={(corner.includes('w') ? shape.x : shape.x + shape.width) - handleR}
                          y={(corner.includes('n') ? shape.y : shape.y + shape.height) - handleR}
                          width={handleR * 2}
                          height={handleR * 2}
                          strokeWidth={1.5 / s}
                          onPointerDown={(e) => startCornerDrag(e, a, corner)}
                        />
                      );
                    })}
                  {selected && tool === 'select' && a.shape.type === 'polygon' &&
                    a.shape.points.map((p, i) => (
                      <circle
                        key={i}
                        className="handle vertex"
                        cx={p[0]}
                        cy={p[1]}
                        r={handleR}
                        strokeWidth={1.5 / s}
                        onPointerDown={(e) => startVertexDrag(e, a, i)}
                      />
                    ))}
                </g>
              );
            })}

            {pending && (
              <g className="pending">
                {pending.polygon && (
                  <polygon
                    className={`shape pending-shape${pending.busy ? ' busy' : ''}`}
                    points={pending.polygon.map((p) => p.join(',')).join(' ')}
                    stroke={activeColor}
                    fill={activeColor}
                  />
                )}
                {pending.box && (
                  <rect className="shape drawing" x={pending.box.x} y={pending.box.y} width={pending.box.width} height={pending.box.height} stroke={activeColor} fill="none" />
                )}
                {pending.points.map((p, i) => (
                  <circle key={i} className={`prompt ${p.positive ? 'positive' : 'negative'}`} cx={p.x} cy={p.y} r={handleR * 1.2} strokeWidth={2 / s} />
                ))}
              </g>
            )}

            {drag?.kind === 'segment' && !drag.exclude && Math.max(Math.abs(drag.current[0] - drag.start[0]), Math.abs(drag.current[1] - drag.start[1])) * s > MIN_BOX_SCREEN_PX * 2 && (() => {
              const b = normalizeBox(drag.start, drag.current);
              return <rect className="shape drawing" x={b.x} y={b.y} width={b.width} height={b.height} stroke={activeColor} fill={activeColor} />;
            })()}

            {drag?.kind === 'box' && (() => {
              const b = normalizeBox(drag.start, drag.current);
              return <rect className="shape drawing" x={b.x} y={b.y} width={b.width} height={b.height} stroke={activeColor} fill={activeColor} />;
            })()}

            {draft.length > 0 && (
              <g className="draft">
                <polyline
                  className="shape drawing"
                  points={[...draft, ...(hover ? [hover] : [])].map((p) => p.join(',')).join(' ')}
                  stroke={activeColor}
                  fill="none"
                />
                {draft.map((p, i) => (
                  <circle key={i} className={`handle vertex${i === 0 && draft.length >= 3 ? ' closable' : ''}`} cx={p[0]} cy={p[1]} r={handleR} strokeWidth={1.5 / s} />
                ))}
              </g>
            )}
          </svg>
        </div>
      )}
      {image && (
        <div className="zoom-controls">
          <button onClick={() => zoomBy(1 / 1.4)} title="Zoom out" aria-label="Zoom out">
            −
          </button>
          <button onClick={fit} title="Fit the image to the window (F)">
            {Math.round(s * 100)}%
          </button>
          <button onClick={() => zoomBy(1.4)} title="Zoom in" aria-label="Zoom in">
            +
          </button>
        </div>
      )}
      {image && (
        <div className="viewer-hud">
          {image.width}×{image.height}
          {tool === 'polygon' && (draft.length === 0 ? ' · Click to add points' : ' · Click the first point, double-click or press Enter to finish')}
          {tool === 'segment' && ` · ${segmentHint(modelStatus, pending)}`}
        </div>
      )}
    </div>
  );
}

function segmentHint(status: ModelStatus, pending: PendingSegment | null) {
  if (status.kind === 'loading') {
    const pct = status.total ? ` ${Math.round((status.loaded / status.total) * 100)}%` : '';
    return `Downloading the segmentation model${pct} (first time only)…`;
  }
  if (status.kind === 'error') return `Segmentation unavailable: ${status.message}`;
  if (!pending) return 'Click an object to segment it, or drag a box around it';
  if (pending.busy) return 'Segmenting…';
  if (pending.error) return pending.error;
  return 'Click this outline to add · right-click to remove · click elsewhere for the next object · Esc to discard';
}
