import { create } from 'zustand';
import type { PromptPoint, Region, SegmentOption } from './segment/protocol';
import { FolderStore, LOCK_STALE_MS, rememberFolder, type ImageEntry, type LockFile } from './project/folder';
import {
  newId,
  newImageDoc,
  newProject,
  nextClassColor,
  type Annotation,
  type BoxShape,
  type ClassDef,
  type ImageDoc,
  type ImageStatus,
  type ProjectFile,
  type Shape,
} from './project/types';

export type Tool = 'segment' | 'select' | 'box' | 'polygon';

/** An object being segmented: prompts so far and the model's latest outline, not yet saved. */
export interface PendingSegment {
  /** Which encoded picture the prompts refer to (image name plus region). */
  key: string;
  region: Region;
  points: PromptPoint[];
  box: BoxShape | null;
  /** The outline being shown: `options[level]`, or null while the first click is still running. */
  polygon: Array<[number, number]> | null;
  /** Readings of the same clicks, biggest first, to step between with ↑ and ↓. */
  options?: SegmentOption[];
  level: number;
  score?: number;
  busy: boolean;
  error: string | null;
}
export type Filter = 'all' | ImageStatus;
export type SaveState = { kind: 'saved' } | { kind: 'pending' } | { kind: 'error'; message: string };
export type OpenResult = { ok: true } | { ok: false; reason: 'empty' } | { ok: false; reason: 'locked'; lock: LockFile };

/** Longest side of the bitmap used for display; zooming past this shows interpolated pixels. */
const MAX_DISPLAY_SIDE = 4096;
const SAVE_DELAY_MS = 700;
const HEARTBEAT_MS = 30_000;
const UNDO_LIMIT = 100;

export interface LoadedImage {
  name: string;
  bitmap: ImageBitmap;
  /** Original (EXIF-oriented) size in pixels; annotation coordinates use this. */
  width: number;
  height: number;
}

interface State {
  folder: FolderStore | null;
  images: ImageEntry[];
  project: ProjectFile;
  filter: Filter;

  currentName: string | null;
  doc: ImageDoc | null;
  image: LoadedImage | null;
  imageError: string | null;

  tool: Tool;
  pendingSegment: PendingSegment | null;
  activeClassId: string | null;
  selectedId: string | null;
  past: Annotation[][];
  future: Annotation[][];
  saveState: SaveState;

  openFolder(handle: FileSystemDirectoryHandle, options?: { force?: boolean }): Promise<OpenResult>;
  closeFolder(): Promise<void>;
  goTo(name: string): Promise<void>;
  goRelative(delta: 1 | -1): Promise<void>;
  setFilter(filter: Filter): void;

  setTool(tool: Tool): void;
  setPendingSegment(pending: PendingSegment | null): void;
  /** Save the pending segmentation as a polygon annotation, if the model produced one. */
  commitPendingSegment(): void;
  setActiveClass(id: string | null): void;
  select(id: string | null): void;

  addAnnotation(shape: Shape, source?: Annotation['source'], score?: number): void;
  /** Change a shape. Pass record: false for intermediate drag updates after calling beginEdit(). */
  updateShape(id: string, shape: Shape, options?: { record?: boolean }): void;
  beginEdit(): void;
  setAnnotationClass(id: string, classId: string | null): void;
  deleteAnnotation(id: string): void;
  undo(): void;
  redo(): void;
  setStatus(status: ImageStatus): void;

  addClass(name: string): ClassDef;
  updateClass(id: string, patch: Partial<Pick<ClassDef, 'name' | 'color'>>): void;
  deleteClass(id: string): void;

  /** Write imported annotations into the project, adding any classes they need. */
  applyImport(docs: ImageDoc[], newClasses: ClassDef[]): Promise<void>;
  flushSaves(): Promise<void>;
}

// ---------------------------------------------------------------------------------------------
// Module-level machinery that doesn't belong in React state: caches, timers, the save queue.

/** Identifies this tab. Kept in sessionStorage so reloading the tab doesn't trip its own folder lock. */
const sessionId = (() => {
  try {
    const existing = sessionStorage.getItem('imagoLabel-session');
    if (existing) return existing;
    const id = newId();
    sessionStorage.setItem('imagoLabel-session', id);
    return id;
  } catch {
    return newId();
  }
})();
/** Docs loaded this session, so quick back-and-forth never races a pending write. */
let docCache = new Map<string, ImageDoc>();
const bitmapCache = new Map<string, Promise<LoadedImage>>();
const dirtyDocs = new Set<string>();
let projectDirty = false;
let saveTimer: ReturnType<typeof setTimeout> | undefined;
let saveChain: Promise<void> = Promise.resolve();
let heartbeat: ReturnType<typeof setInterval> | undefined;
let navToken = 0;

const now = () => new Date().toISOString();

async function decodeImage(entry: ImageEntry): Promise<LoadedImage> {
  const file = await entry.handle.getFile();
  const full = await createImageBitmap(file, { imageOrientation: 'from-image' });
  const { width, height } = full;
  const scale = Math.min(1, MAX_DISPLAY_SIDE / Math.max(width, height));
  if (scale === 1) return { name: entry.name, bitmap: full, width, height };
  const bitmap = await createImageBitmap(full, {
    resizeWidth: Math.round(width * scale),
    resizeHeight: Math.round(height * scale),
    resizeQuality: 'high',
  });
  full.close();
  return { name: entry.name, bitmap, width, height };
}

/** Decoded image for any image in the open folder (cached for the current image and its neighbours). */
export function loadImageByName(name: string) {
  const entry = useStore.getState().images.find((i) => i.name === name);
  return entry ? loadImage(entry) : Promise.reject(new Error(`No image named ${name}`));
}

function loadImage(entry: ImageEntry) {
  let pending = bitmapCache.get(entry.name);
  if (!pending) {
    pending = decodeImage(entry);
    pending.catch(() => bitmapCache.delete(entry.name));
    bitmapCache.set(entry.name, pending);
  }
  return pending;
}

/** Keep decoded bitmaps only for the current image and its neighbours. */
function trimBitmaps(keep: Set<string>) {
  for (const [name, pending] of bitmapCache) {
    if (keep.has(name)) continue;
    bitmapCache.delete(name);
    pending.then((img) => img.bitmap.close()).catch(() => undefined);
  }
}

/** Drop references to classes that no longer exist (e.g. deleted while this image wasn't loaded). */
function normalizeDoc(doc: ImageDoc, classes: ClassDef[]): ImageDoc {
  const ids = new Set(classes.map((c) => c.id));
  if (doc.annotations.every((a) => a.classId === null || ids.has(a.classId))) return doc;
  return { ...doc, annotations: doc.annotations.map((a) => (a.classId && !ids.has(a.classId) ? { ...a, classId: null } : a)) };
}

export const useStore = create<State>()((set, get) => {
  function scheduleSave() {
    set({ saveState: { kind: 'pending' } });
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => void get().flushSaves(), SAVE_DELAY_MS);
  }

  function markProjectDirty(project: ProjectFile) {
    projectDirty = true;
    set({ project: { ...project, updatedAt: now() } });
    scheduleSave();
  }

  /** Replace the current image's annotations, keeping status, the project index and undo history in sync. */
  function commitAnnotations(annotations: Annotation[], options: { record?: boolean; status?: ImageStatus } = {}) {
    const { doc, project, past } = get();
    if (!doc) return;
    let status = options.status ?? doc.status;
    if (!options.status) {
      if (status === 'todo' && annotations.length > 0) status = 'in-progress';
      if (status === 'in-progress' && annotations.length === 0) status = 'todo';
    }
    const next: ImageDoc = { ...doc, annotations, status, updatedAt: now() };
    docCache.set(next.image.name, next);
    dirtyDocs.add(next.image.name);

    const images = { ...project.images };
    if (status === 'todo' && annotations.length === 0) delete images[next.image.name];
    else images[next.image.name] = { status, annotations: annotations.length, updatedAt: next.updatedAt };

    const record = options.record ?? true;
    set({
      doc: next,
      past: record ? [...past, doc.annotations].slice(-UNDO_LIMIT) : past,
      future: record ? [] : get().future,
    });
    markProjectDirty({ ...get().project, images });
  }

  return {
    folder: null,
    images: [],
    project: newProject(),
    filter: 'all',
    currentName: null,
    doc: null,
    image: null,
    imageError: null,
    tool: 'select',
    pendingSegment: null,
    activeClassId: null,
    selectedId: null,
    past: [],
    future: [],
    saveState: { kind: 'saved' },

    async openFolder(handle, options = {}) {
      const folder = new FolderStore(handle);
      const images = await folder.listImages();
      if (images.length === 0) return { ok: false, reason: 'empty' };
      await folder.migrateLegacyData();

      const lock = await folder.readLock().catch(() => null);
      const lockIsLive = lock && lock.sessionId !== sessionId && Date.now() - Date.parse(lock.heartbeat) < LOCK_STALE_MS;
      if (lockIsLive && !options.force) return { ok: false, reason: 'locked', lock };

      await get().closeFolder();
      const project = (await folder.readProject()) ?? newProject();
      docCache = new Map();
      set({
        folder,
        images,
        project,
        filter: 'all',
        activeClassId: project.classes[0]?.id ?? null,
        saveState: { kind: 'saved' },
      });

      const writeLock = () =>
        folder.writeLock({ sessionId, openedAt: lock?.sessionId === sessionId ? lock.openedAt : now(), heartbeat: now() }).catch(() => undefined);
      await writeLock();
      heartbeat = setInterval(writeLock, HEARTBEAT_MS);
      void rememberFolder(handle);

      const start = images.find((i) => i.name === project.lastImage) ?? images.find((i) => !project.images[i.name] || project.images[i.name].status === 'todo') ?? images[0];
      await get().goTo(start.name);
      return { ok: true };
    },

    async closeFolder() {
      const { folder } = get();
      if (!folder) return;
      await get().flushSaves();
      clearInterval(heartbeat);
      await folder.removeLock(sessionId).catch(() => undefined);
      trimBitmaps(new Set());
      docCache = new Map();
      set({ folder: null, images: [], project: newProject(), currentName: null, doc: null, image: null, selectedId: null, past: [], future: [] });
    },

    async goTo(name) {
      const { folder, images, project, currentName } = get();
      if (!folder) return;
      const index = images.findIndex((i) => i.name === name);
      if (index < 0) return;
      get().commitPendingSegment();
      const token = ++navToken;
      if (currentName !== name) void get().flushSaves();
      set({ currentName: name, selectedId: null, past: [], future: [], imageError: null });

      try {
        const [image, stored] = await Promise.all([
          loadImage(images[index]),
          docCache.has(name) ? Promise.resolve(docCache.get(name)!) : folder.readImageDoc(name),
        ]);
        if (token !== navToken) return;
        let doc = stored ? normalizeDoc(stored, get().project.classes) : newImageDoc(name, image.width, image.height);
        if (doc.image.width !== image.width || doc.image.height !== image.height) {
          doc = { ...doc, image: { name, width: image.width, height: image.height } };
        }
        docCache.set(name, doc);
        set({ doc, image });
        if (project.lastImage !== name) markProjectDirty({ ...get().project, lastImage: name });
      } catch (err) {
        if (token !== navToken) return;
        set({ doc: null, image: null, imageError: err instanceof Error ? err.message : String(err) });
      }

      // Decode neighbours in the background so moving to them is instant.
      const neighbours = [images[index - 1], images[index + 1]].filter(Boolean) as ImageEntry[];
      trimBitmaps(new Set([name, ...neighbours.map((n) => n.name)]));
      for (const n of neighbours) void loadImage(n).catch(() => undefined);
    },

    async goRelative(delta) {
      const { images, currentName, filter, project } = get();
      const start = images.findIndex((i) => i.name === currentName);
      for (let i = start + delta; i >= 0 && i < images.length; i += delta) {
        const status = project.images[images[i].name]?.status ?? 'todo';
        if (filter === 'all' || status === filter) return get().goTo(images[i].name);
      }
    },

    setFilter: (filter) => set({ filter }),
    setTool(tool) {
      if (tool !== 'segment') get().commitPendingSegment();
      set({ tool, selectedId: tool === 'select' ? get().selectedId : null });
    },

    setPendingSegment: (pendingSegment) => set({ pendingSegment }),

    commitPendingSegment() {
      const { pendingSegment } = get();
      if (!pendingSegment) return;
      set({ pendingSegment: null });
      if (pendingSegment.polygon && pendingSegment.polygon.length >= 3) {
        get().addAnnotation({ type: 'polygon', points: pendingSegment.polygon }, 'sam', pendingSegment.score);
      }
    },
    setActiveClass: (activeClassId) => set({ activeClassId }),
    select: (selectedId) => set({ selectedId }),

    addAnnotation(shape, source = 'manual', score) {
      const { doc, activeClassId } = get();
      if (!doc) return;
      const time = now();
      const annotation: Annotation = { id: newId(), classId: activeClassId, shape, source, score, createdAt: time, updatedAt: time };
      commitAnnotations([...doc.annotations, annotation]);
      set({ selectedId: annotation.id });
    },

    beginEdit() {
      const { doc, past } = get();
      if (doc) set({ past: [...past, doc.annotations].slice(-UNDO_LIMIT), future: [] });
    },

    updateShape(id, shape, options = {}) {
      const { doc } = get();
      if (!doc) return;
      commitAnnotations(
        doc.annotations.map((a) => (a.id === id ? { ...a, shape, updatedAt: now() } : a)),
        { record: options.record },
      );
    },

    setAnnotationClass(id, classId) {
      const { doc } = get();
      if (!doc) return;
      commitAnnotations(doc.annotations.map((a) => (a.id === id ? { ...a, classId, updatedAt: now() } : a)));
    },

    deleteAnnotation(id) {
      const { doc, selectedId } = get();
      if (!doc) return;
      commitAnnotations(doc.annotations.filter((a) => a.id !== id));
      if (selectedId === id) set({ selectedId: null });
    },

    undo() {
      const { doc, past, future } = get();
      if (!doc || past.length === 0) return;
      const previous = past[past.length - 1];
      commitAnnotations(previous, { record: false });
      set({ past: past.slice(0, -1), future: [doc.annotations, ...future], selectedId: null });
    },

    redo() {
      const { doc, past, future } = get();
      if (!doc || future.length === 0) return;
      commitAnnotations(future[0], { record: false });
      set({ past: [...past, doc.annotations], future: future.slice(1), selectedId: null });
    },

    setStatus(status) {
      const { doc } = get();
      if (doc) commitAnnotations(doc.annotations, { record: false, status });
    },

    addClass(name) {
      const { project } = get();
      const cls: ClassDef = { id: newId(), name: name.trim(), color: nextClassColor(project.classes) };
      markProjectDirty({ ...project, classes: [...project.classes, cls] });
      if (!get().activeClassId) set({ activeClassId: cls.id });
      return cls;
    },

    updateClass(id, patch) {
      const { project } = get();
      markProjectDirty({ ...project, classes: project.classes.map((c) => (c.id === id ? { ...c, ...patch } : c)) });
    },

    deleteClass(id) {
      const { project, doc, activeClassId } = get();
      const classes = project.classes.filter((c) => c.id !== id);
      markProjectDirty({ ...project, classes });
      if (activeClassId === id) set({ activeClassId: classes[0]?.id ?? null });
      for (const [name, cached] of docCache) docCache.set(name, normalizeDoc(cached, classes));
      if (doc && doc.annotations.some((a) => a.classId === id)) {
        commitAnnotations(doc.annotations.map((a) => (a.classId === id ? { ...a, classId: null } : a)), { record: false });
      }
    },

    async applyImport(docs, newClasses) {
      const { project, currentName } = get();
      const images = { ...project.images };
      for (const doc of docs) {
        docCache.set(doc.image.name, doc);
        dirtyDocs.add(doc.image.name);
        images[doc.image.name] = { status: doc.status, annotations: doc.annotations.length, updatedAt: doc.updatedAt };
      }
      markProjectDirty({ ...project, classes: [...project.classes, ...newClasses], images });
      if (!get().activeClassId) set({ activeClassId: get().project.classes[0]?.id ?? null });
      const current = docs.find((d) => d.image.name === currentName);
      if (current) set({ doc: current, past: [], future: [], selectedId: null });
      await get().flushSaves();
    },

    flushSaves() {
      clearTimeout(saveTimer);
      saveChain = saveChain.then(async () => {
        const { folder } = get();
        if (!folder || (dirtyDocs.size === 0 && !projectDirty)) return;
        // Clear each dirty flag before writing, so edits made during the write mark it dirty again.
        let writing: string | 'project' | null = null;
        try {
          for (const name of [...dirtyDocs]) {
            writing = name;
            dirtyDocs.delete(name);
            const doc = docCache.get(name);
            if (doc) await folder.writeImageDoc(doc);
          }
          if (projectDirty) {
            writing = 'project';
            projectDirty = false;
            await folder.writeProject(get().project);
          }
          if (dirtyDocs.size === 0 && !projectDirty) set({ saveState: { kind: 'saved' } });
        } catch (err) {
          if (writing === 'project') projectDirty = true;
          else if (writing) dirtyDocs.add(writing);
          set({ saveState: { kind: 'error', message: err instanceof Error ? err.message : String(err) } });
          // Keep retrying: the usual causes (network drive hiccup, sync client holding the file) are temporary.
          saveTimer = setTimeout(() => void get().flushSaves(), 5000);
        }
      });
      return saveChain;
    },
  };
});

export const hasUnsavedChanges = () => dirtyDocs.size > 0 || projectDirty;
