function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function preferredEmbedHeight(width, mode) {
  const viewportHeight = window.innerHeight || 900;
  const modeFloor = {
    inverted_pendulum: 680,
    localization: 700,
    path_planning: 720,
    slam: 720,
    robot: 780,
  };

  const minHeight = (modeFloor[mode] || 700) + (width < 760 ? 80 : 0);
  let preferred;

  if (width < 760) {
    preferred = width * 1.18;
  } else if (width < 1100) {
    preferred = width * 0.9;
  } else {
    preferred = width * 0.66;
  }

  if (mode === "robot") {
    preferred += 50;
  }

  const maxHeight = Math.max(minHeight, Math.min(viewportHeight * 0.9, 980));
  return Math.round(clamp(preferred, minHeight, maxHeight));
}

function updateEmbedFrame(frame) {
  if (frame.dataset.heightReported === "1") {
    return;
  }

  const width = frame.getBoundingClientRect().width;
  if (!width) {
    return;
  }

  const mode = frame.dataset.simMode || "default";
  frame.style.height = `${preferredEmbedHeight(width, mode)}px`;
}

function bindEmbedFrame(frame) {
  const update = () => updateEmbedFrame(frame);
  frame.dataset.heightReported = "";
  frame.dataset.maxReportedHeight = "";
  update();

  if (typeof ResizeObserver !== "undefined") {
    const target = frame.parentElement || frame;
    const observer = new ResizeObserver(update);
    observer.observe(target);
  }

  window.addEventListener("resize", update, { passive: true });
}

window.addEventListener("message", (event) => {
  const data = event.data;
  if (!data || data.type !== "rust-robotics-embed-size" || typeof data.height !== "number") {
    return;
  }

  for (const frame of document.querySelectorAll(".sim-embed-frame")) {
    const reportedHeight = Math.round(clamp(data.height, 480, 1600));
    const previousReportedHeight =
      Number.parseFloat(frame.dataset.maxReportedHeight || "0") || 0;
    const height = Math.max(previousReportedHeight, reportedHeight);
    frame.dataset.maxReportedHeight = `${height}`;
    frame.dataset.heightReported = "1";
    frame.style.height = `${height}px`;
  }
});

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".sim-embed-frame").forEach(bindEmbedFrame);
});
