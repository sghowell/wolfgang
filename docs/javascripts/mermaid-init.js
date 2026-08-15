(() => {
  const mermaid = window.mermaid;
  if (!mermaid) {
    return;
  }

  const themeName = () =>
    document.body?.getAttribute("data-md-color-scheme") === "slate" ? "dark" : "default";

  const resetContainer = (container) => {
    const source = container.dataset.mermaidSource ?? container.textContent ?? "";
    container.dataset.mermaidSource = source;
    container.removeAttribute("data-processed");
    container.textContent = source.trim();
  };

  const promoteFence = (node) => {
    if (!(node instanceof HTMLElement) || node.dataset.mermaidPromoted === "true") {
      return node;
    }
    const source = node.textContent ?? "";
    node.dataset.mermaidPromoted = "true";
    node.dataset.mermaidSource = source;
    node.textContent = source.trim();
    return node;
  };

  const collectContainers = (root) => {
    const scope = root instanceof Document ? root.body ?? root : root;
    if (!(scope instanceof ParentNode)) {
      return [];
    }
    const promoted = Array.from(scope.querySelectorAll("pre.mermaid")).map(promoteFence);
    const existing = Array.from(scope.querySelectorAll(".mermaid[data-mermaid-source]"));
    return [...new Set([...promoted, ...existing])];
  };

  const renderMermaid = async (root = document) => {
    const containers = collectContainers(root);
    if (!containers.length) {
      return;
    }

    mermaid.initialize({
      startOnLoad: false,
      securityLevel: "strict",
      theme: themeName(),
    });

    for (const container of containers) {
      resetContainer(container);
    }

    await mermaid.run({ nodes: containers });
  };

  const scheduleRender = (root = document) => {
    queueMicrotask(() => {
      void renderMermaid(root);
    });
  };

  document$.subscribe((root) => {
    scheduleRender(root);
  });

  document.addEventListener("DOMContentLoaded", () => {
    scheduleRender(document);
  });

  const observer = new MutationObserver(() => {
    scheduleRender(document);
  });

  observer.observe(document.body, {
    attributes: true,
    attributeFilter: ["data-md-color-scheme"],
  });
})();
