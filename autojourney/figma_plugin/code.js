// AutoJourney publisher plugin — runs in Figma's main plugin sandbox, which
// has full document/scene-graph access but no network access. The UI iframe
// (ui.html) fetches the journey spec + screen images from the local
// autojourney CLI process and hands them over via postMessage.

figma.showUI(__html__, { visible: false });

function post(type, payload) {
  figma.ui.postMessage(Object.assign({ type }, payload || {}));
}

figma.ui.onmessage = async (msg) => {
  if (msg.type === "fetch-error") {
    figma.notify("AutoJourney: could not reach local server — " + msg.error, { error: true });
    figma.closePlugin();
    return;
  }
  if (msg.type !== "build") return;

  try {
    await buildJourney(msg.spec, msg.images);
    post("complete", { success: true });
    figma.notify("AutoJourney: journey map published");
  } catch (err) {
    const detail = (err && err.message) || String(err);
    post("complete", { success: false, error: detail });
    figma.notify("AutoJourney failed: " + detail, { error: true });
  }
  figma.closePlugin();
};

async function buildJourney(spec, images) {
  const total = spec.screens.length;
  let done = 0;

  let page = figma.root.children.find((p) => p.name === spec.pageName);
  if (!page) {
    page = figma.createPage();
    page.name = spec.pageName;
  }
  await figma.setCurrentPageAsync(page);

  await figma.loadFontAsync({ family: "Inter", style: "Regular" });
  await figma.loadFontAsync({ family: "Inter", style: "Bold" });

  const nodeIdByScreenId = {};
  for (const s of spec.screens) {
    const frame = figma.createFrame();
    frame.name = s.name;
    frame.resize(s.w, s.h);
    frame.x = s.x;
    frame.y = s.y;

    const bytes = new Uint8Array(images[s.id] || []);
    if (bytes.length > 0) {
      const image = figma.createImage(bytes);
      frame.fills = [{ type: "IMAGE", imageHash: image.hash, scaleMode: "FILL" }];
    } else {
      frame.fills = [{ type: "SOLID", color: { r: 0.94, g: 0.94, b: 0.94 } }];
    }
    page.appendChild(frame);

    const label = figma.createText();
    label.fontName = { family: "Inter", style: "Bold" };
    label.characters = s.label;
    label.fontSize = 13;
    label.resize(s.w, label.height);
    label.x = s.x;
    label.y = s.y + s.h + 8;
    page.appendChild(label);

    if (s.action) {
      const actionText = figma.createText();
      actionText.fontName = { family: "Inter", style: "Regular" };
      actionText.characters = "↳ " + s.action;
      actionText.fontSize = 11;
      actionText.resize(s.w, actionText.height);
      actionText.x = s.x;
      actionText.y = s.y + s.h + 28;
      page.appendChild(actionText);
    }

    nodeIdByScreenId[s.id] = frame.id;
    done += 1;
    post("progress", { done, total, detail: "Placed: " + (s.label || s.id) });
  }

  for (const e of spec.edges) {
    const fromId = nodeIdByScreenId[e.fromId];
    const toId = nodeIdByScreenId[e.toId];
    if (!fromId || !toId) continue;
    const fromNode = await figma.getNodeByIdAsync(fromId);
    const toNode = await figma.getNodeByIdAsync(toId);
    if (!fromNode || !toNode) continue;

    const startX = fromNode.x + fromNode.width / 2;
    const startY = fromNode.y + fromNode.height;
    const endX = toNode.x + toNode.width / 2;
    const endY = toNode.y;

    const minX = Math.min(startX, endX);
    const minY = Math.min(startY, endY);
    const w = Math.max(Math.abs(endX - startX), 1);
    const h = Math.max(Math.abs(endY - startY), 1);

    const vector = figma.createVector();
    vector.resize(w, h);
    vector.x = minX;
    vector.y = minY;
    vector.vectorPaths = [
      {
        windingRule: "NONE",
        data: "M " + (startX - minX) + " " + (startY - minY) + " L " + (endX - minX) + " " + (endY - minY),
      },
    ];
    vector.strokeWeight = 2;
    vector.strokes = [{ type: "SOLID", color: { r: 0.35, g: 0.4, b: 0.9 } }];
    vector.strokeCap = "ARROW_LINES";
    page.appendChild(vector);

    if (e.label) {
      const text = figma.createText();
      text.fontName = { family: "Inter", style: "Regular" };
      text.characters = e.label;
      text.fontSize = 10;
      text.x = (startX + endX) / 2;
      text.y = (startY + endY) / 2;
      page.appendChild(text);
    }
  }

  figma.viewport.scrollAndZoomIntoView(page.children);
}
