const nonce = new URLSearchParams(location.hash.slice(1)).get("n") || "";
history.replaceState(null, "", location.pathname);
const status = document.getElementById("status");
const error = document.getElementById("error");
async function signIn() {
  if (!nonce) { status.textContent = ""; error.textContent = "No sign-in code. Launch PhaseEval again to open a signed-in window."; return; }
  const response = await fetch("/api/launcher/exchange", {
    method: "POST", cache: "no-store", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ nonce }),
  });
  if (!response.ok) {
    status.textContent = "";
    error.textContent = "This sign-in code was already used or has expired. Launch PhaseEval again to open a signed-in window.";
    return;
  }
  const body = await response.json();
  try {
    for (const key of ["axa.owner_token", "axa.dashboard.token", "phaseeval.rater_token", "phaseeval.admin_token"]) sessionStorage.removeItem(key);
    for (const [key, value] of Object.entries(body.session_storage)) sessionStorage.setItem(key, value);
  }
  catch (e) { error.textContent = "This browser blocks session storage, so PhaseEval cannot keep you signed in."; return; }
  location.replace(body.next);
}
signIn();
