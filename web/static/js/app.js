(function () {
  var zone = document.getElementById("dropzone");
  if (!zone) return;
  var input = document.getElementById("file-input"), form = document.getElementById("upload-form");
  var analyze = document.getElementById("analyze"), choose = document.getElementById("choose");
  var preview = document.getElementById("preview"), dropText = document.getElementById("drop-text");
  var nameEl = document.getElementById("file-name"), maxMb = parseFloat(zone.dataset.maxMb || "5");

  function setFile(file) {
    if (!file) return;
    if (!/^image\/(png|jpeg)$/.test(file.type)) { nameEl.textContent = "Please choose a PNG or JPG image."; analyze.disabled = true; return; }
    var dt = new DataTransfer(); dt.items.add(file); input.files = dt.files;
    nameEl.textContent = file.name + " (" + (file.size / 1024).toFixed(0) + " KB)";
    preview.src = URL.createObjectURL(file); preview.hidden = false; dropText.hidden = true; analyze.disabled = false;
  }
  choose.addEventListener("click", function () { input.click(); });
  zone.addEventListener("click", function () { input.click(); });
  zone.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  input.addEventListener("change", function () { setFile(input.files[0]); });
  ["dragenter", "dragover"].forEach(function (ev) { zone.addEventListener(ev, function (e) { e.preventDefault(); zone.classList.add("over"); }); });
  ["dragleave", "drop"].forEach(function (ev) { zone.addEventListener(ev, function (e) { e.preventDefault(); zone.classList.remove("over"); }); });
  zone.addEventListener("drop", function (e) { setFile(e.dataTransfer.files[0]); });
  form.addEventListener("submit", function () { analyze.disabled = true; analyze.textContent = "Analyzing…"; });
})();
