function carsSwitchTab(button, type) {
  const container = button.closest(".cars-tabs");

  // Switch active button
  container.querySelectorAll(".cars-tab-btn").forEach(btn => btn.classList.remove("active"));
  button.classList.add("active");

  // Switch content
  container.querySelectorAll(".cars-tab-content").forEach(c => c.classList.remove("active"));
  container.querySelector(`.cars-tab-content.${type}`).classList.add("active");
}

function carsCopyCode(button) {
  const container = button.closest(".cars-tabs");
  const activeCode = container.querySelector(".cars-tab-content.active pre.highlight");
  if (!activeCode) return;

  navigator.clipboard.writeText(activeCode.innerText).then(() => {
    button.textContent = "✔ Done";
    setTimeout(() => { button.textContent = "Copy"; }, 2000);
  });
}

function carsCopyFull(button) {
    // trouver le container .cars-tabs le plus proche
    const container = button.closest(".cars-tabs");
    if (!container) return;

    // déterminer l’onglet actif
    const activeTab = container.querySelector(".cars-tab-content.active");
    if (!activeTab) return;

    // récupérer le texte complet à copier
    let fullText = "";
    if (activeTab.classList.contains("yaml")) {
        fullText = container.dataset.fullYaml;   // stocké côté Python
    } else if (activeTab.classList.contains("json")) {
        fullText = container.dataset.fullJson;   // stocké côté Python
    }

    navigator.clipboard.writeText(fullText).then(() => {
        const oldText = button.textContent;
        button.textContent = "✔ Done";
        setTimeout(() => { button.textContent = oldText; }, 2000);
    });
}
