import { Modal } from './Modal';

const GROUPS: Array<{ title: string; rows: Array<[string, string]> }> = [
  {
    title: 'Tools',
    rows: [
      ['S', 'Segment: click an object to outline it'],
      ['V', 'Select: move and edit shapes'],
      ['B', 'Draw a box'],
      ['P', 'Draw a polygon'],
    ],
  },
  {
    title: 'Segmenting',
    rows: [
      ['Click', 'Outline the object you clicked'],
      ['Click on or near it', 'Add a part it missed'],
      ['Ctrl+click', 'Add a part further away, without starting a new object'],
      ['Right-click', 'Remove an area from the outline'],
      ['Drag', 'Outline whatever is inside the box you drag'],
      ['Backspace', 'Undo your last click'],
      ['Enter', 'Keep the outline'],
      ['Esc', 'Discard the outline'],
    ],
  },
  {
    title: 'Drawing and editing',
    rows: [
      ['Drag a shape', 'Move it (Select tool)'],
      ['Drag a handle', 'Resize a box or move a polygon point'],
      ['Alt+click a point', 'Remove that polygon point'],
      ['1 – 9', 'Use that class, and apply it to the selected shape'],
      ['Double-click a class', 'Rename it (or use its pencil button)'],
      ['Delete', 'Delete the selected shape'],
      ['Ctrl+Z / Ctrl+Shift+Z', 'Undo / redo'],
    ],
  },
  {
    title: 'Moving around',
    rows: [
      ['← / →', 'Previous / next image'],
      ['Enter', 'Mark the image done and go to the next'],
      ['Scroll', 'Zoom in and out'],
      ['Space+drag, middle-drag', 'Pan'],
      ['F', 'Fit the image to the window'],
      ['?', 'Show this list'],
    ],
  },
];

export function ShortcutsDialog({ onClose }: { onClose: () => void }) {
  return (
    <Modal title="Keyboard and mouse" onClose={onClose}>
      {GROUPS.map((group) => (
        <section key={group.title} className="shortcuts">
          <h3>{group.title}</h3>
          <dl>
            {group.rows.map(([keys, what]) => (
              <div key={keys}>
                <dt>
                  {keys.split(' / ').map((key, i) => (
                    <span key={key}>
                      {i > 0 && <span className="muted"> / </span>}
                      <kbd>{key}</kbd>
                    </span>
                  ))}
                </dt>
                <dd>{what}</dd>
              </div>
            ))}
          </dl>
        </section>
      ))}
      <div className="modal-actions">
        <button className="primary" onClick={onClose}>
          Close
        </button>
      </div>
    </Modal>
  );
}
