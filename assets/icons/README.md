# UI Icons Folder

Icons have been generated and integrated.

## Icons List (v2/assets/icons/)

```
analytics.svg    # Bar chart - EDA tab
brain.svg       # Neural network - Architecture
clipboard.svg   # Clipboard - Layers tab
clipboard-outline.svg  # Clipboard outline
document.svg   # Document - Form input
ruler.svg       # Ruler - Evaluation tab
search.svg     # Magnifying glass - Predict
settings.svg   # Gear - Tuning
shield.svg     # Shield - Brand logo
trending-up.svg # Line chart - Training
alert.svg      # Triangle - Warning
check.svg      # Checkmark - Verified
circle-red.svg    # Red circle - High risk
circle-orange.svg # Orange circle - Medium risk
circle-green.svg # Green circle - Low risk
```

## Already Integrated in HTML

- Favicon: shield.svg
- Sidebar brand: shield.svg
- Navigation tabs: 6 icons (analytics, brain, trending-up, settings, ruler, search)
- Card headers: document, clipboard, ruler, brain, analytics
- Risk badges: circle-red, circle-orange, circle-green

## CSS Updates

- Updated `.sidebar__icon`, `.header__icon`, `.sidebar__item-icon`, `.card__header-icon` to use `width/height` instead of `font-size`

## How to Use

Drop new SVG files here → reference in HTML:
```html
<img class="icon" src="{{ url_for('static', filename='icons/name.svg') }}" alt="Icon">