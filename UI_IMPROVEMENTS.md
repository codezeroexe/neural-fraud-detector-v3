# UI/UX Improvements Checklist — v2

> Monochrome design system inspired by Claude UI
> Batch implementation plan for incremental improvements

---

## Batches Overview

| Batch | Focus | Files |
|-------|-------|-------|
| 1 | Design Tokens — Monochrome | `static/styles.css` |
| 2 | Layout — Sidebar Navigation | `templates/index.html`, `styles.css` |
| 3 | Component Styling | `static/styles.css` |
| 4 | Micro-interactions | `static/styles.css` |
| 5 | Loading & Empty States | `templates/index.html` |
| 6 | Accessibility & Keyboard | `templates/index.html` |
| 7 | Advanced Features | `templates/index.html`, `app.py` |

---

## Design Tokens (Batch 1: Foundation)

### Monochrome Color System

#### Dark Mode
```css
--bg-primary: #0A0A0A;      /* Deep black */
--bg-card: #141414;            /* Slightly lighter */
--bg-card-hover: #1A1A1A;      /* Hover state */
--bg-input: #1A1A1A;         /* Input fields */
--text-primary: #EDEDED;       /* High contrast white */
--text-secondary: #A1A1AA;   /* Muted gray */
--text-muted: #71717A;        /* Very muted */
--border: #262626;             /* Subtle zinc border */
--accent: #FFFFFF;            /* White for dark mode */
--focus-ring: rgba(255, 255, 255, 0.25);
--active-state: #262626;        /* Active tab bg */
```

#### Light Mode
```css
--bg-primary: #FFFFFF;        /* Pure white */
--bg-card: #F5F5F5;           /* Slightly darker */
--bg-card-hover: #EBEBEB;     /* Hover state */
--bg-input: #F5F5F5;         /* Input fields */
--text-primary: #171717;      /* Near black */
--text-secondary: #737373;     /* Muted gray */
--text-muted: #A1A1AA;        /* Very muted */
--border: #E5E5E5;            /* Light border */
--accent: #171717;            /* Black for light mode */
--focus-ring: rgba(0, 0, 0, 0.25);
--active-state: #E5E5E5;       /* Active tab bg */
```

### Typography
- Font family: Inter
- Monospace: JetBrains Mono (stats/numbers)
- Base size: 14px
- Line height: 1.5 (body), 1.2 (headings)

### Spacing Scale
- Base unit: 4px
- Scale: 4, 8, 12, 16, 24, 32, 48, 64px

### Border Radius
- Cards: 12px
- Buttons: 8px
- Inputs: 8px

---

## Batch 1: Design Tokens Checklist

```markdown
## Batch 1: Design Tokens — Monochrome

### Color System
- [x] Dark mode background (#0A0A0A)
- [x] Dark card background (#141414)
- [x] Dark card hover (#1A1A1A)
- [x] Dark text primary (#EDEDED)
- [x] Dark text secondary (#A1A1AA)
- [x] Dark border (#262626)
- [x] Dark accent (#FFFFFF)
- [x] Dark focus ring (white 25%)
- [x] Dark active state (#262626 bg)

- [x] Light mode background (#FFFFFF)
- [x] Light card background (#F5F5F5)
- [x] Light card hover (#EBEBEB)
- [x] Light text primary (#171717)
- [x] Light text secondary (#737373)
- [x] Light border (#E5E5E5)
- [x] Light accent (#171717)
- [x] Light focus ring (black 25%)
- [x] Light active state (#E5E5E5 bg)

### Typography
- [x] Inter font (already using)
- [x] JetBrains Mono for stats/numbers
- [x] Base font size 14px
- [x] Line height 1.5 body, 1.2 headings

### Spacing & Radius
- [x] Consistent spacing scale (4px base)
- [x] Border radius 12px cards
- [x] Border radius 8px buttons
```

---

## Batch 2: Layout — Sidebar Navigation

### Structure
```html
<!-- Sidebar Layout -->
<div class="app-layout">
  <aside class="sidebar">
    <div class="sidebar__brand">🛡️ Fraud Detection</div>
    <nav class="sidebar__nav">
      <button class="sidebar__item active" data-tab="eda">
        <icon>📊</icon>
        <label>EDA Analysis</label>
      </button>
      <!-- ... more tabs ... -->
    </nav>
  </aside>
  <main class="main-content">
    <!-- Tab panels -->
  </main>
</div>
```

### CSS Structure
```css
.app-layout {
  display: flex;
  min-height: 100vh;
}

.sidebar {
  width: 240px;
  position: fixed;
  left: 0;
  top: 0;
  bottom: 0;
  overflow-y: auto;
}

.main-content {
  margin-left: 240px;
  flex: 1;
}

/* Collapsed state */
.sidebar.collapsed {
  width: 64px;
}
.sidebar.collapsed .sidebar__item label {
  display: none;
}
```

### Batch 2 Checklist

```markdown
## Batch 2: Layout — Sidebar Navigation

### Structure
- [x] Convert top tabs to left sidebar
- [x] Sidebar width 240px
- [x] Brand/logo at top
- [x] Collapsed state 64px icon-only
- [x] Toggle button for collapse

### Active States
- [x] Active tab: accent bg
- [x] Active tab: white/black text
- [x] Hover: subtle bg change

### Mobile
- [x] Hamburger menu button
- [x] Drawer overlay on mobile
- [x] Close on outside click
```

---

## Batch 3: Component Styling

### Button Variants
```css
.btn {
  padding: 10px 16px;
  border-radius: 8px;
  font-weight: 500;
  transition: all 150ms ease;
}

.btn--primary {
  background: var(--accent);
  color: var(--bg-primary);
}

.btn--secondary {
  background: transparent;
  border: 1px solid var(--border);
}

.btn--ghost {
  background: transparent;
}
```

### Input Focus Glow
```css
.input:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--focus-ring);
}
```

### Batch 3 Checklist

```markdown
## Batch 3: Component Styling

### Buttons
- [ ] Primary button (accent bg)
- [ ] Secondary button (border only)
- [ ] Ghost button (no border)
- [ ] Hover lift + glow
- [ ] Disabled state styling

### Inputs
- [ ] Focus glow ring
- [ ] Border color change on focus
- [ ] Placeholder styling

### Cards
- [ ] Border styling
- [ ] Elevation/shadow
- [ ] Hover lift effect

### Stats & Tables
- [ ] Stats card styling
- [ ] Table header styling
- [ ] Table row hover
```

---

## Batch 4: Micro-interactions

### Animations
```css
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes slideIn {
  from { transform: translateX(-16px); opacity: 0; }
  to { transform: translateX(0); opacity: 1; }
}

.tab-panel.active {
  animation: fadeIn 200ms ease-out;
}

/* Staggered list items */
.list-item:nth-child(1) { animation-delay: 0ms; }
.list-item:nth-child(2) { animation-delay: 50ms; }
.list-item:nth-child(3) { animation-delay: 100ms; }
```

### Batch 4 Checklist

```markdown
## Batch 4: Micro-interactions

### Focus
- [ ] Focus ring animation
- [ ] Focus visible states

### Transitions
- [ ] Page transitions (200ms)
- [ ] Tab switch animation
- [ ] Card hover transitions

### Animations
- [ ] Staggered list items
- [ ] Result reveal animation
- [ ] Count-up animation for stats

### Motion Preferences
- [ ] Reduced motion support
- [ ] prefers-reduced-motion media query
```

---

## Batch 5: Loading & Empty States

### Skeleton Loader
```css
.skeleton {
  background: linear-gradient(
    90deg,
    var(--bg-card) 25%,
    var(--bg-card-hover) 50%,
    var(--bg-card) 75%
  );
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
}
```

### Batch 5 Checklist

```markdown
## Batch 5: Loading & Empty States

### Loading
- [ ] Skeleton loading states
- [ ] Button spinner
- [ ] Chart placeholders

### Empty States
- [ ] No prediction history
- [ ] No tuning results
- [ ] No training history
```

---

## Batch 6: Accessibility

### Keyboard Navigation
```javascript
// Arrow key navigation between tabs
document.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowRight') {
    // Next tab
  } else if (e.key === 'ArrowLeft') {
    // Previous tab
  }
});
```

### Batch 6 Checklist

```markdown
## Batch 6: Accessibility & Keyboard

### Keyboard
- [ ] Tab/Arrow key navigation
- [ ] Escape to close
- [ ] Number keys 1-6 for tabs
- [ ] Focus trap in modals

### A11y
- [ ] Skip links
- [ ] ARIA live regions
- [ ] Screen reader support
- [ ] aria-label on icons
```

---

## Batch 7: Advanced Features

### Toast Notifications
```javascript
function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast toast--${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}
```

### Batch 7 Checklist

```markdown
## Batch 7: Advanced Features

### Notifications
- [ ] Toast container
- [ ] Success toasts
- [ ] Error toasts
- [ ] Auto-dismiss (3s)

### Features
- [ ] Prediction history (localStorage)
- [ ] Keyboard shortcuts help (? key)
- [ ] Chart export (PNG)
- [ ] Threshold slider
```

---

## Files to Modify

| Batch | Files Modified |
|-------|---------------|
| 1 | `static/styles.css` |
| 2 | `templates/index.html`, `static/styles.css` |
| 3 | `static/styles.css` |
| 4 | `static/styles.css` |
| 5 | `templates/index.html`, `static/styles.css` |
| 6 | `templates/index.html`, `static/styles.css` |
| 7 | `templates/index.html`, `app.py` |

---

## Completion Priority

1. **Batch 1** — Foundation (visible immediately)
2. **Batch 2** — Sidebar (Claude-like structure)
3. **Batch 3** — Component polish
4. **Batch 4** — Micro-interactions
5. **Batch 5** — Loading states
6. **Batch 6** — Accessibility
7. **Batch 7** — Advanced features

> Each batch builds on the previous. After Batch 2, the UI will have a completely different feel (Claude-inspired).