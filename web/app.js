(() => {
  const modelInput = document.querySelector("#model");
  const prompt = document.querySelector("#prompt");
  const generate = document.querySelector("#generate");
  const cancel = document.querySelector("#cancel");
  const output = document.querySelector("#output");
  let runtimeReady = window.miniinferRuntimeReady === true;
  let handle = 0;

  const refresh = () => { generate.disabled = !runtimeReady || !handle; };
  window.miniinferReady = () => { runtimeReady = true; refresh(); };

  modelInput.addEventListener("change", async () => {
    if (!runtimeReady || !modelInput.files.length) return;
    if (handle) Module.ccall("miniinfer_destroy", null, ["number"], [handle]);
    const bytes = new Uint8Array(await modelInput.files[0].arrayBuffer());
    try { Module.FS.unlink("/model.miniinfer"); } catch (_) {}
    Module.FS.writeFile("/model.miniinfer", bytes);
    handle = Module.ccall("miniinfer_create", "number",
      ["string", "number", "number"], ["/model.miniinfer", 1, 0]);
    if (!handle) output.textContent = "Could not load this model.";
    refresh();
  });

  generate.addEventListener("click", () => {
    output.textContent = prompt.value;
    generate.disabled = true;
    cancel.disabled = false;
    const callback = Module.addFunction((_token, piece) => {
      output.textContent += Module.UTF8ToString(piece);
      return 1;
    }, "iiii");
    setTimeout(() => {
      const result = Module.ccall("miniinfer_generate", "number",
        ["number", "string", "number", "number", "number", "number"],
        [handle, prompt.value, 32, 1, callback, 0]);
      if (result < 0) output.textContent += "\n[Generation failed]";
      Module.removeFunction?.(callback);
      generate.disabled = false;
      cancel.disabled = true;
    }, 0);
  });

  cancel.addEventListener("click", () => {
    Module.ccall("miniinfer_cancel", null, ["number"], [handle]);
  });
})();
