(function () {
  function resolve(mode) {
    if (mode === "light" || mode === "dark") return mode;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function getStoredMode() {
    try {
      var raw = window.localStorage.getItem("theme-mode");
      if (!raw) return "system";
      var parsed = JSON.parse(raw);
      return parsed || "system";
    } catch (e) {
      return "system";
    }
  }

  function apply(theme) {
    document.documentElement.setAttribute("data-theme", theme);
  }

  // Aplicar de inmediato para evitar un parpadeo con el tema equivocado
  // mientras Dash termina de montar la app.
  apply(resolve(getStoredMode()));

  // Si el modo es "automático", seguir los cambios de tema del sistema
  // operativo mientras la pestaña sigue abierta.
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
    var mode = getStoredMode();
    if (mode === "system") {
      var resolved = resolve("system");
      apply(resolved);
      if (window.dash_clientside && window.dash_clientside.set_props) {
        window.dash_clientside.set_props("theme-resolved", { data: resolved });
      }
    }
  });
})();
