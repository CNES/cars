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