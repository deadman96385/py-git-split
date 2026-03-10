#!/usr/bin/env python3
import os, sys, re, json, subprocess, tempfile, time
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import click

from repo_analysis import analyze_repo, RepoAnalysis, RepoMode
from structural_partitioning import partition, PartitionResult
from feature_extraction import extract_features
from candidate_bundling import build_bundles, generate_export_report, Bundle, MAX_ANCHOR_BREADTH
from embedding_refinement import refine_bundles

try:
    import questionary
except Exception:
    questionary = None
try:
    import anthropic
except Exception:
    anthropic = None
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None
try:
    from rich.console import Console
    from rich.table import Table
except Exception:
    Console = None
    Table = None

console = Console() if Console else None

@dataclass
class Hunk:
    id:str; filepath:str; header:str; lines:list[str]; index:int
    @property
    def added(self): return sum(1 for l in self.lines if l.startswith('+'))
    @property
    def removed(self): return sum(1 for l in self.lines if l.startswith('-'))

@dataclass
class FileHeader:
    filepath:str; lines:list[str]

@dataclass
class CommitGroup:
    id:str; label:str; message:str; hunk_ids:list[str]=field(default_factory=list); body:Optional[str]=None; file_paths:list[str]=field(default_factory=list)




def _load_env_file(path):
    p = Path(path)
    if not p.exists() or not p.is_file():
        return
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        t = line.strip()
        if not t or t.startswith("#") or "=" not in t:
            continue
        k, v = t.split("=", 1)
        k = k.strip()
        if not k:
            continue
        v = v.strip()
        if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
            v = v[1:-1]
        os.environ.setdefault(k, v)


def load_env_defaults():
    # Load from current working directory first, then script directory.
    cwd_env = Path.cwd() / ".env"
    script_env = Path(__file__).resolve().parent / ".env"
    if load_dotenv:
        load_dotenv(dotenv_path=cwd_env, override=False)
        if script_env != cwd_env:
            load_dotenv(dotenv_path=script_env, override=False)
        return
    _load_env_file(cwd_env)
    _load_env_file(script_env)


def _env_flag(name, default=True):
    raw = os.environ.get(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


def status(msg):
    if console:
        console.print(f"[cyan][git-split][/cyan] {msg}")
        return
    print(f"[git-split] {msg}", flush=True)
def run_git(*args, check=True, capture=True):
    return subprocess.run(["git",*args], capture_output=capture, text=True, encoding="utf-8", errors="replace", check=check)


def run_git_env(*args, env=None, check=True, capture=True):
    return subprocess.run(["git",*args], env=env, capture_output=capture, text=True, encoding="utf-8", errors="replace", check=check)


class _GitOut:
    """Minimal stand-in for CompletedProcess when we need binary→str decoding."""
    __slots__ = ("stdout", "stderr", "returncode")
    def __init__(self, stdout: str, stderr: str, returncode: int = 0):
        self.stdout = stdout; self.stderr = stderr; self.returncode = returncode


def run_git_diff(*args, check=True) -> _GitOut:
    """Run git in raw-bytes mode, decode with surrogateescape.

    Using text=True would invoke Python's universal-newlines translation which
    silently converts \\r\\n → \\n (and bare \\r → \\n).  That strips the \\r
    from CRLF-format context lines, so the reconstructed patch no longer matches
    the indexed content and git apply reports 'corrupt patch'.  Binary mode +
    manual decode is lossless.
    """
    r = subprocess.run(["git", *args], capture_output=True, check=check)
    if check and r.returncode:
        raise subprocess.CalledProcessError(r.returncode, ["git", *args], r.stdout, r.stderr)
    return _GitOut(
        r.stdout.decode("utf-8", errors="surrogateescape"),
        r.stderr.decode("utf-8", errors="replace"),
        r.returncode,
    )


def run_git_env_diff(*args, env=None, check=True) -> _GitOut:
    """Like run_git_diff but with a custom environment."""
    r = subprocess.run(["git", *args], env=env, capture_output=True, check=check)
    if check and r.returncode:
        raise subprocess.CalledProcessError(r.returncode, ["git", *args], r.stdout, r.stderr)
    return _GitOut(
        r.stdout.decode("utf-8", errors="surrogateescape"),
        r.stderr.decode("utf-8", errors="replace"),
        r.returncode,
    )


def expected_tree_from_patch(raw_patch, base_ref="HEAD"):
    status(f"computing expected tree from patch (base={base_ref})")
    if not raw_patch.strip():
        return run_git("rev-parse", f"{base_ref}^{{tree}}" if base_ref != "HEAD" else "HEAD^{tree}").stdout.strip()
    pf = tempfile.NamedTemporaryFile("wb", suffix=".diff", delete=False, dir=tempfile.gettempdir())
    idxf = tempfile.NamedTemporaryFile("wb", suffix=".idx", delete=False, dir=tempfile.gettempdir())
    try:
        pf.write(raw_patch.encode("utf-8", errors="surrogateescape"))
        patch_path = pf.name
    finally:
        pf.close()
    try:
        idx_path = idxf.name
    finally:
        idxf.close()
    env = os.environ.copy()
    env["GIT_INDEX_FILE"] = idx_path
    try:
        run_git_env("read-tree", base_ref, env=env)
        run_git_env_diff("apply", "--cached", "--whitespace=nowarn", patch_path, env=env)
        tree = run_git_env("write-tree", env=env).stdout.strip()
        status(f"expected tree computed: {tree[:12]}")
        return tree
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"failed to build expected tree from patch: {e.stderr or e}")
    finally:
        for n in (patch_path, idx_path):
            try:
                os.unlink(n)
            except OSError:
                pass


def verify_tree_or_die(expected_tree, context_label):
    status(f"verifying final tree ({context_label})")
    actual = run_git("rev-parse", "HEAD^{tree}").stdout.strip()
    if actual != expected_tree:
        raise RuntimeError(f"{context_label}: tree mismatch expected={expected_tree} actual={actual}")

def assert_repo():
    try: run_git("rev-parse","--git-dir")
    except subprocess.CalledProcessError:
        print("Error: not inside git repo", file=sys.stderr); sys.exit(1)


def assert_clean_state():
    """Abort early if git is mid-operation or working tree is dirty."""
    git_dir = run_git("rev-parse","--git-dir").stdout.strip()
    in_progress = []
    for marker in ("REBASE_HEAD", "MERGE_HEAD", "CHERRY_PICK_HEAD", "BISECT_LOG", "rebase-merge", "rebase-apply"):
        if os.path.exists(os.path.join(git_dir, marker)):
            in_progress.append(marker)
    if in_progress:
        print(f"Error: git operation in progress ({', '.join(in_progress)}). Abort or finish it first.", file=sys.stderr)
        sys.exit(1)
    # Check for uncommitted changes (index or worktree)
    result = run_git("status","--porcelain","--untracked-files=no")
    dirty = [l for l in result.stdout.splitlines() if l.strip()]
    if dirty:
        print("Error: working tree has uncommitted changes. Stash or commit them first:", file=sys.stderr)
        for l in dirty[:10]:
            print(f"  {l}", file=sys.stderr)
        if len(dirty) > 10:
            print(f"  ... and {len(dirty)-10} more", file=sys.stderr)
        sys.exit(1)

def _strip_binary_diff_sections(raw: str) -> tuple[str, int]:
    """Remove binary-only file sections from a unified diff.

    git apply cannot handle 'Binary files X and Y differ' entries without the
    actual binary content (--binary flag on git show). Since those blobs already
    exist in the object store (they're part of a committed revision), they can be
    staged separately — we don't need to validate them via git apply.

    Returns (stripped_diff, binary_file_count).
    """
    out: list[str] = []
    section: list[str] = []
    section_is_binary = False
    binary_count = 0

    for line in raw.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if section:
                if section_is_binary:
                    binary_count += 1
                else:
                    out.extend(section)
            section = [line]
            section_is_binary = False
        else:
            if line.startswith("Binary files ") or line.startswith("GIT binary patch"):
                section_is_binary = True
            section.append(line)

    if section:
        if section_is_binary:
            binary_count += 1
        else:
            out.extend(section)

    return "".join(out), binary_count


def preflight_patch_check(raw_diff: str, parent_ref: str) -> None:
    """Verify the full diff applies cleanly against parent_ref using a temp index.

    Binary-only file entries are stripped before testing — they can't be applied
    without --binary content, but their blobs are already in the object store.

    Aborts with sys.exit(1) if git apply would fail, so we don't waste API credits
    on a diff that can never be committed. The real repo index is never touched.
    """
    text_diff, binary_count = _strip_binary_diff_sections(raw_diff)
    msg = f"pre-flight: verifying text hunks apply cleanly against {parent_ref[:12]}"
    if binary_count:
        msg += f" ({binary_count} binary-only files skipped)"
    status(msg)

    if not text_diff.strip():
        status("pre-flight: no text hunks to validate")
        return

    pf = tempfile.NamedTemporaryFile("wb", suffix="_preflight.diff", delete=False,
                                     dir=tempfile.gettempdir())
    idxf = tempfile.NamedTemporaryFile("wb", suffix="_preflight.idx", delete=False,
                                       dir=tempfile.gettempdir())
    try:
        pf.write(text_diff.encode("utf-8", errors="surrogateescape"))
        patch_path = pf.name
    finally:
        pf.close()
    try:
        idx_path = idxf.name
    finally:
        idxf.close()
    env = os.environ.copy()
    env["GIT_INDEX_FILE"] = idx_path
    try:
        run_git_env("read-tree", parent_ref, env=env)
        run_git_env_diff("apply", "--cached", "--whitespace=nowarn", patch_path, env=env)
        status("pre-flight: patch applies cleanly")
    except subprocess.CalledProcessError as e:
        err_text = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        print("", file=sys.stderr)
        print("ERROR: pre-flight patch check failed — aborting before any API calls.", file=sys.stderr)
        print(err_text.rstrip(), file=sys.stderr)
        # Show context around the corrupt line to aid diagnosis
        m = re.search(r'corrupt patch at line (\d+)', err_text)
        if m:
            bad = int(m.group(1))
            diff_lines = text_diff.split("\n")
            lo, hi = max(0, bad - 5), min(len(diff_lines), bad + 4)
            print(f"\nContext around diff line {bad}:", file=sys.stderr)
            for i, l in enumerate(diff_lines[lo:hi], lo + 1):
                marker = ">>>" if i == bad else "   "
                print(f"  {marker} {i:6d}: {repr(l)}", file=sys.stderr)
        sys.exit(1)
    finally:
        for n in (patch_path, idx_path):
            try:
                os.unlink(n)
            except OSError:
                pass


def get_diff(staged=True):
    if staged:
        return run_git_diff("diff", "--cached", "--unified=3").stdout
    return run_git_diff("diff", "--unified=3").stdout

def get_commit_diff(h): return run_git_diff("show","--unified=3",h).stdout

def get_commit_msg(h): return run_git("log","--format=%B","-n","1",h).stdout.strip()

def parse_diff(raw):
    fh, hunks = {}, []
    fr = re.compile(r'^diff --git a/(.*?) b/(.*?)$')
    hr = re.compile(r'^@@[^@]*@@.*$')
    fp, fhl, hh, hl, idx = None, [], None, [], defaultdict(int)
    def flush_hunk():
        nonlocal hh, hl
        if hh and fp:
            i = idx[fp]; hunks.append(Hunk(f"{fp}::{i}",fp,hh,list(hl),i)); idx[fp]=i+1
        hh, hl = None, []
    def flush_fh():
        nonlocal fhl
        if fp and fhl: fh[fp]=FileHeader(fp,list(fhl))
    # Use split('\n') not splitlines() — splitlines() strips \r from CRLF lines,
    # which corrupts context lines when the patch is reconstructed and applied.
    raw_lines = raw.split('\n')
    if raw_lines and raw_lines[-1] == '':
        raw_lines = raw_lines[:-1]
    for line in raw_lines:
        s = line.rstrip('\r')  # strip trailing \r for matching only; line is stored as-is
        m = fr.match(s)
        if m:
            flush_hunk(); flush_fh(); fp=m.group(2); fhl=[line]; hh=None; hl=[]; continue
        if fp and any(s.startswith(p) for p in ("index ","--- ","+++ ","new file","deleted file","Binary","old mode","new mode","rename ","similarity ")):
            if hh is None: fhl.append(line)
            continue
        if hr.match(s):
            flush_hunk(); hh=line; hl=[]; continue
        if hh is not None: hl.append(line)
    flush_hunk(); flush_fh(); return fh,hunks

def build_patch(file_headers,hunks,hids):
    sel=set(hids); out=[]; seen=[]; by=defaultdict(list)
    for h in hunks:
        if h.id in sel:
            if h.filepath not in by: seen.append(h.filepath)
            by[h.filepath].append(h)
    for fp in seen:
        if fp in file_headers: out += file_headers[fp].lines
        for h in by[fp]: out.append(h.header); out += h.lines
    return "\n".join(out)+"\n" if out else ""

def stage_patch(p):
    with tempfile.NamedTemporaryFile("wb", suffix=".patch", delete=False) as f:
        f.write(p.encode("utf-8", errors="surrogateescape")); n=f.name
    try: run_git("apply","--cached","--whitespace=nowarn",n); return True
    except subprocess.CalledProcessError as e: print(e.stderr,file=sys.stderr); return False
    finally:
        try: os.unlink(n)
        except OSError: pass


def stage_files(file_paths):
    lf = tempfile.NamedTemporaryFile("w", suffix=".lst", delete=False, dir=tempfile.gettempdir())
    try:
        for fp in file_paths:
            lf.write(fp + "\n")
        list_path = lf.name
    finally:
        lf.close()
    try:
        run_git("add", "-A", f"--pathspec-from-file={list_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(e.stderr, file=sys.stderr)
        return False
    finally:
        try:
            os.unlink(list_path)
        except OSError:
            pass

def _is_new(h): return any(x.startswith("new file") for x in (h.lines if h else []))

def _require_anthropic(api_key):
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for semantic split modes")
    if not anthropic:
        raise RuntimeError("anthropic package is required for semantic split modes")


_RATE_LIMIT_MAX_ATTEMPTS = 6
_RATE_LIMIT_INITIAL_WAIT = 60   # seconds; doubled each attempt, capped at 300


def _api_call_with_retry(fn, label="API call"):
    """Call fn() with exponential backoff on rate limit (429) errors.

    Reads retry-after from the error response headers when available,
    otherwise falls back to doubling wait time each attempt.

    Returns (result, total_waited_seconds).
    """
    wait = _RATE_LIMIT_INITIAL_WAIT
    total_waited = 0.0
    for attempt in range(_RATE_LIMIT_MAX_ATTEMPTS):
        try:
            return fn(), total_waited
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = (
                "rate_limit" in err_str
                or "rate limit" in err_str
                or "429" in str(e)
                or (anthropic and isinstance(e, getattr(anthropic, "RateLimitError", type(None))))
            )
            if not is_rate_limit or attempt == _RATE_LIMIT_MAX_ATTEMPTS - 1:
                raise
            retry_after = None
            try:
                if hasattr(e, "response") and hasattr(e.response, "headers"):
                    ra = e.response.headers.get("retry-after") or e.response.headers.get("x-ratelimit-reset-requests")
                    if ra:
                        retry_after = int(float(ra))
            except Exception:
                pass
            actual_wait = retry_after if retry_after is not None else wait
            status(f"{label}: rate limited, waiting {actual_wait}s (attempt {attempt + 1}/{_RATE_LIMIT_MAX_ATTEMPTS})...")
            time.sleep(actual_wait)
            total_waited += actual_wait
            wait = min(wait * 2, 300)




def _normalize_subject(msg):
    m=re.match(r'^([a-z]+)\(([^)]+)\):\s*(.+)$', (msg or '').strip())
    if m:
        typ, scope, rest = m.groups()
        scope = scope.replace('-', ' ').strip()
        return f"{typ}: {scope}: {rest.strip()}"
    return (msg or '').strip()


def _clean_bullet_line(line):
    t=str(line).strip()
    # prevent "- - ..." or "* - ..." duplicates
    while True:
        nt=re.sub(r'^(?:[-*]\s+)+', '', t)
        if nt == t:
            break
        t = nt.strip()
    return t


def _fallback_subject_for_group(files, added, removed, fhs=None):
    if not files:
        return "chore: repo: update metadata"
    scopes=defaultdict(int)
    for fp in files:
        parts = fp.replace("\\", "/").split("/")
        s = parts[0] if parts else "repo"
        scopes[s] += 1
    scope=sorted(scopes.items(), key=lambda x:(-x[1], x[0]))[0][0].replace('-', ' ')

    all_new=False
    if fhs is not None:
        all_new=all(_is_new(fhs.get(fp)) for fp in files)

    if all_new:
        typ, verb = "feat", "import"
    elif added > 0 and removed == 0:
        typ, verb = "feat", "add"
    elif added == 0 and removed > 0:
        typ, verb = "fix", "remove"
    else:
        typ, verb = "chore", "update"
    return f"{typ}: {scope}: {verb} {len(files)} files"


_MSG_BATCH_SIZE = 20          # groups per Claude request
_MSG_MAX_FILES = 12           # file paths per group entry
_MSG_MAX_PREVIEW_HUNKS = 4    # hunks to include in preview per group
_MSG_MAX_PREVIEW_LINES = 8    # diff lines per preview hunk
_MSG_MAX_LINE_CHARS = 180     # truncate long diff lines
_MSG_MAX_CONTEXT_CHARS = 800  # truncate the original/bundle-context string
_MSG_INTER_BATCH_DELAY = 4    # seconds between batches


def ai_generate_messages(groups, hunks, api_key, original=None, body_mode="auto", fhs=None):
    status(f"AI message generation start: {len(groups)} groups (body_mode={body_mode})")
    _require_anthropic(api_key)
    c=anthropic.Anthropic(api_key=api_key)
    hmap={h.id:h for h in hunks}
    # Truncate context string so it doesn't dominate token budget
    original_trunc = (original or "")[:_MSG_MAX_CONTEXT_CHARS]
    payload=[]
    group_meta={}
    for g in groups:
        hs=[hmap[hid] for hid in g.hunk_ids if hid in hmap]
        files=sorted(set({h.filepath for h in hs}) | set(g.file_paths or []))
        added=sum(h.added for h in hs)
        removed=sum(h.removed for h in hs)
        new_files=sum(1 for fp in files if _is_new(fhs.get(fp))) if fhs is not None else 0
        file_only_count=max(0, len(files) - len({h.filepath for h in hs}))
        group_meta[g.id] = {"files": files, "added": added, "removed": removed}
        preview = [
            {
                "file": h.filepath,
                "header": h.header[:80],
                "sample": "\n".join(
                    ln[:_MSG_MAX_LINE_CHARS] for ln in h.lines[:_MSG_MAX_PREVIEW_LINES]
                ),
            }
            for h in hs[:_MSG_MAX_PREVIEW_HUNKS]
        ]
        payload.append({
            "id": g.id,
            "files": files[:_MSG_MAX_FILES],
            "file_count": len(files),
            "new_files": new_files,
            "file_only_count": file_only_count,
            "hunks": len(hs),
            "added": added,
            "removed": removed,
            "preview": preview,
        })

    # Send in batches to stay within token rate limits
    by: dict = {}
    total_batches = (len(payload) + _MSG_BATCH_SIZE - 1) // _MSG_BATCH_SIZE
    for batch_idx in range(total_batches):
        batch = payload[batch_idx * _MSG_BATCH_SIZE : (batch_idx + 1) * _MSG_BATCH_SIZE]
        if total_batches > 1:
            status(f"AI message generation: batch {batch_idx + 1}/{total_batches} ({len(batch)} groups)...")
        req={
            "original": original_trunc,
            "body_mode": body_mode,
            "commits": batch,
            "format": "Return JSON only: {commits:[{id,subject,body_lines:[...]}]} subject <= 100 chars, format strictly as type: scope: summary (example: fix: kernel: remove unused keys). Never say no file changes when file_count > 0.",
        }
        _req = req  # capture for lambda
        r, waited = _api_call_with_retry(
            lambda: c.messages.create(model="claude-sonnet-4-20250514", max_tokens=8192, system="Return JSON only.", messages=[{"role": "user", "content": json.dumps(_req)}]),
            label=f"message generation batch {batch_idx + 1}/{total_batches}",
        )
        raw=r.content[0].text.strip()
        raw=re.sub(r"^```[a-z]*\n?","",raw); raw=re.sub(r"\n?```$","",raw)
        raw=raw.strip()
        start=raw.find('{'); end=raw.rfind('}')
        if start != -1 and end != -1 and end > start:
            raw=raw[start:end+1]
        try:
            d=json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"AI response (raw):\n{raw}", file=sys.stderr)
            raise RuntimeError(f"AI returned invalid JSON for message generation (batch {batch_idx+1}): {e}") from e
        by.update({x.get("id"):x for x in d.get("commits",[])})
        if batch_idx < total_batches - 1:
            extra = max(0.0, _MSG_INTER_BATCH_DELAY - waited)
            if extra > 0:
                time.sleep(extra)

    for g in groups:
        x=by.get(g.id)
        if not x or not x.get("subject"):
            raise RuntimeError(f"message generation missing subject for group {g.id}")
        msg=_normalize_subject(x["subject"])
        gm=group_meta.get(g.id, {"files":[],"added":0,"removed":0})
        if gm["files"] and re.search(r"no\s+file\s+change", msg, flags=re.IGNORECASE):
            msg=_fallback_subject_for_group(gm["files"], gm["added"], gm["removed"], fhs=fhs)
        g.message=msg
        bl=[_clean_bullet_line(z) for z in x.get("body_lines",[]) if _clean_bullet_line(z)]
        if body_mode=="off":
            g.body=None
        elif body_mode=="always" and not bl:
            raise RuntimeError(f"message body required but missing for group {g.id}")
        else:
            g.body="\n".join(f"- {z}" for z in bl[:5]) if bl else None
    status("AI message generation done")
    return groups

_GROUPS_MAX_PREVIEW_LINES = 6
_GROUPS_MAX_LINE_CHARS = 140
_GROUPS_MAX_CONTEXT_CHARS = 800

_MERGE_BATCH_SIZE = 150   # max low-confidence groups per merge-review batch


def _ai_merge_groups(groups: list, bundle_map: dict, api_key: str, context: str = "") -> list:
    """Ask Claude which low-confidence groups should be merged, using bundle summaries.

    Works at the group level (summaries only, no raw diffs) so it scales to any
    number of groups.  Returns a new list of CommitGroups with merges applied.
    Groups not mentioned in the merge response are returned unchanged.
    """
    _require_anthropic(api_key)
    c = anthropic.Anthropic(api_key=api_key)

    # Build compact summary rows — each is ~100-200 tokens
    rows = []
    for g in groups:
        b = bundle_map.get(g.id)
        if b:
            rows.append({"id": g.id, "summary": b.summary()})
        else:
            fps = sorted(set(hid.split("::")[0] for hid in g.hunk_ids))
            rows.append({"id": g.id, "summary": f"{g.label} — {len(g.hunk_ids)} hunks in {', '.join(fps[:5])}"})

    prompt = (
        (context[:_GROUPS_MAX_CONTEXT_CHARS] + "\n\n" if context else "") +
        "The following groups were produced by a structural kernel diff analyser with LOW confidence.\n"
        "Identify groups that are clearly about the same topic/feature and should be merged into one commit.\n"
        "Do NOT merge groups that belong to different subsystems or features, even if they are close in the tree.\n"
        "Leave single-file or narrow groups as-is unless there is an obvious match.\n"
        "If a group's Subsystem field lists more than 3 unrelated top-level subsystems "
        "(e.g. net/, drivers/acpi, drivers/crypto), treat it as a candidate for further splitting "
        "rather than merging — do NOT merge it with other groups.\n\n"
        "Groups:\n" +
        "\n\n".join(f'[{r["id"]}]\n{r["summary"]}' for r in rows) +
        '\n\nReturn JSON only: {"merges": [["id_a", "id_b"], ["id_c", "id_d", "id_e"]]}\n'
        "Only include groups that should be merged. Omit groups that should stay as-is."
    )

    result, _ = _api_call_with_retry(
        lambda: c.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system="Return JSON only.",
            messages=[{"role": "user", "content": prompt}],
        ),
        label="group merge review",
    )
    raw = result.content[0].text.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw); raw = re.sub(r"\n?```$", "", raw)
    s = raw.find("{"); e = raw.rfind("}")
    if s == -1 or e <= s:
        status("merge review returned no valid JSON — groups unchanged")
        return groups
    try:
        data = json.loads(raw[s:e+1])
    except json.JSONDecodeError:
        status("merge review JSON parse failed — groups unchanged")
        return groups

    merges: list[list[str]] = data.get("merges", [])
    if not merges:
        return groups

    # Apply merges: for each merge set, fold everything into the first group
    id_to_group = {g.id: g for g in groups}
    absorbed: set[str] = set()
    for merge_set in merges:
        valid = [gid for gid in merge_set if gid in id_to_group and gid not in absorbed]
        if len(valid) < 2:
            continue
        primary_id = valid[0]
        primary = id_to_group[primary_id]
        for gid in valid[1:]:
            other = id_to_group[gid]
            primary.hunk_ids = list(primary.hunk_ids) + list(other.hunk_ids)
            primary.file_paths = sorted(set(list(primary.file_paths) + list(other.file_paths)))
            absorbed.add(gid)
        status(f"merge review: merged {valid[1:]} → {primary_id}")

    result_groups = [g for g in groups if g.id not in absorbed]
    status(f"merge review: {len(groups)} → {len(result_groups)} groups ({len(absorbed)} absorbed)")
    return result_groups


def ai_groups(hunks,api_key,msg=None):
    status(f"AI grouping start: {len(hunks)} units")
    _require_anthropic(api_key)
    c=anthropic.Anthropic(api_key=api_key)
    arr=[{
        "id": h.id,
        "file": h.filepath,
        "hunk_header": h.header[:80],
        "preview": "\n".join(ln[:_GROUPS_MAX_LINE_CHARS] for ln in h.lines[:_GROUPS_MAX_PREVIEW_LINES]),
        "added": h.added,
        "removed": h.removed,
    } for h in hunks]
    req={
        "original": (msg or "")[:_GROUPS_MAX_CONTEXT_CHARS],
        "commits": arr,
        "format": "Return JSON only: {commits:[{label,message,hunk_ids,confidence}]}",
    }
    _req = req
    r, _waited = _api_call_with_retry(
        lambda: c.messages.create(model="claude-sonnet-4-20250514", max_tokens=8192, system="Return JSON only.", messages=[{"role": "user", "content": json.dumps(_req)}]),
        label="AI grouping",
    )
    raw=r.content[0].text.strip()
    raw=re.sub(r"^```[a-z]*\n?","",raw); raw=re.sub(r"\n?```$","",raw)
    raw=raw.strip()
    start=raw.find('{'); end=raw.rfind('}')
    if start != -1 and end != -1 and end > start:
        raw=raw[start:end+1]
    try:
        d=json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"AI response (raw):\n{raw}", file=sys.stderr)
        raise RuntimeError(f"AI returned invalid JSON for grouping: {e}") from e
    ids={h.id for h in hunks}; out=[]
    for i,x in enumerate(d.get("commits",[])):
        v=[z for z in x.get("hunk_ids",[]) if z in ids]
        if not v:
            continue
        out.append(CommitGroup(f"group-{i}",x.get("label",f"Commit {i+1}"),x.get("message","chore: changes"),v))
    status("AI grouping response received; verifying integrity")
    ok,msgv=verify_integrity(hunks,out)
    if not ok:
        raise RuntimeError(f"AI grouping failed integrity: {msgv}")
    status(f"AI grouping done: {len(out)} groups")
    return out

def _prune_empty_groups(groups, hunks=None):
    hmap = {h.id: h for h in (hunks or [])}
    out=[]
    for g in groups:
        has_files=bool(g.file_paths)
        if not has_files and g.hunk_ids:
            has_files=any(hid in hmap for hid in g.hunk_ids) if hunks is not None else True
        if has_files:
            out.append(g)
    return out


def verify_integrity(hunks, groups):
    all_ids = {h.id for h in hunks}
    assigned = {}
    dup = []
    for g in groups:
        for hid in g.hunk_ids:
            if hid in assigned:
                dup.append(hid)
            assigned[hid] = g.id
    miss = sorted(all_ids - set(assigned.keys()))
    extra = sorted(set(assigned.keys()) - all_ids)
    if not miss and not dup and not extra:
        return True, "ok"
    return False, f"missing={len(miss)} dup={len(dup)} extra={len(extra)}"


def _print_groups(groups, hunks):
    hmap = {h.id: h for h in hunks}
    if console and Table:
        table = Table(title="Proposed commits", show_header=True, header_style="bold")
        table.add_column("#", justify="right")
        table.add_column("Message")
        table.add_column("Stats")
        for i, g in enumerate(groups, 1):
            added = removed = 0
            files = set(g.file_paths or [])
            resolved_hunks = 0
            for hid in g.hunk_ids:
                h = hmap.get(hid)
                if h:
                    resolved_hunks += 1
                    files.add(h.filepath)
                    added += h.added
                    removed += h.removed
            stats = f"{resolved_hunks} hunks, {len(files)} files, +{added}/-{removed}"
            table.add_row(str(i), g.message, stats)
        console.print()
        console.print(table)
        return
    print("\nProposed commits:")
    for i, g in enumerate(groups, 1):
        added = removed = 0
        files = set(g.file_paths or [])
        resolved_hunks = 0
        for hid in g.hunk_ids:
            h = hmap.get(hid)
            if h:
                resolved_hunks += 1
                files.add(h.filepath)
                added += h.added
                removed += h.removed
        print(f"  [{i}] {g.message}  ({resolved_hunks} hunks, {len(files)} files, +{added}/-{removed})")

def _choose(prompt, choices):
    if questionary:
        q = [questionary.Choice(label, value=val) for label, val in choices]
        return questionary.select(prompt, choices=q).ask()
    print(prompt)
    for i, (label, _) in enumerate(choices, 1):
        print(f"  {i}. {label}")
    try:
        n = int(input("> ").strip())
        if 1 <= n <= len(choices):
            return choices[n-1][1]
        return None
    except Exception:
        return None


def interactive_edit(groups, hunks, fhs=None, api_key=None, original_message=None):
    auto_accept = _env_flag("GIT_SPLIT_AUTO_ACCEPT", default=False)
    if auto_accept:
        ok, msg = verify_integrity(hunks, groups)
        if not ok:
            raise RuntimeError(f"Integrity error before auto-accept: {msg}")
        status("interactive review auto-accepted (GIT_SPLIT_AUTO_ACCEPT=true)")
        return groups

    if not questionary:
        ok, msg = verify_integrity(hunks, groups)
        if not ok:
            print(f"Integrity warning: {msg}")
        return groups

    while True:
        _print_groups(groups, hunks)
        action = _choose(
            "What next?",
            [
                ("Accept", "accept"),
                ("Move hunk", "move"),
                ("Edit commit message", "edit"),
                ("Re-run AI grouping", "rerun"),
                ("Abort", "abort"),
            ],
        )
        if action in (None, "abort"):
            print("Aborted.")
            sys.exit(0)
        if action == "accept":
            ok, msg = verify_integrity(hunks, groups)
            if ok:
                return groups
            print(f"Integrity error: {msg}")
            continue
        if action == "move":
            hid = _choose(
                "Select hunk",
                [(f"{h.filepath} :: {h.header[:60]}", h.id) for h in hunks],
            )
            if not hid:
                continue
            gid = _choose(
                "Move to commit",
                [(f"[{i+1}] {g.message}", g.id) for i, g in enumerate(groups)],
            )
            if not gid:
                continue
            for g in groups:
                if hid in g.hunk_ids:
                    g.hunk_ids.remove(hid)
            target = next((g for g in groups if g.id == gid), None)
            if target is None:
                continue
            target.hunk_ids.append(hid)
            groups = _prune_empty_groups(groups, hunks=hunks)
            continue
        if action == "edit":
            gid = _choose(
                "Which commit message?",
                [(f"[{i+1}] {g.message}", g.id) for i, g in enumerate(groups)],
            )
            if not gid:
                continue
            g = next((x for x in groups if x.id == gid), None)
            if g is None:
                continue
            new_msg = questionary.text("New message:", default=g.message).ask()
            if new_msg:
                g.message = new_msg.strip()
            continue
        if action == "rerun":
            groups = ai_groups(hunks, api_key, msg=original_message)
            groups = ai_generate_messages(groups, hunks, api_key, original=original_message, body_mode="auto", fhs=fhs)
            groups = _prune_empty_groups(groups, hunks=hunks)
            continue


def _auto_rebase(full_hash, parent, script_path):
    short = full_hash[:7]
    seq = f"""#!/usr/bin/env python3\nimport re,sys\np=open(sys.argv[1]).read()\np=re.sub(r'^pick {short}', 'edit {short}', p, flags=re.MULTILINE)\nopen(sys.argv[1],'w').write(p)\n"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=tempfile.gettempdir()) as f:
        f.write(seq)
        sp = f.name
    os.chmod(sp, 0o755)
    env = os.environ.copy()
    env["GIT_SEQUENCE_EDITOR"] = sp
    env["GIT_EDITOR"] = "true"
    try:
        subprocess.run(["git", "rebase", "-i", parent], env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Auto rebase failed: {e}", file=sys.stderr)
        print(f"Run manually: bash {sp}")
        return
    subprocess.run(["bash", script_path], check=True)
def _commit_group(g):
    args=["commit","-m",g.message]
    if g.body:
        args += ["-m",g.body]
    run_git(*args)

def execute_staged(groups,hunks,fhs,expected_tree,dry=False):
    groups = _prune_empty_groups(groups, hunks=hunks)
    status(f"execute staged: {len(groups)} commits")
    if dry:
        for i,g in enumerate(groups,1):
            extra = f" | {g.body}" if g.body else ""
            print(f"[{i}] {g.message}{extra} ({len(g.hunk_ids)} hunks)")
        return
    status("WARNING: About to reset the git index (unstage all). Ctrl-C to cancel.")
    run_git("reset","HEAD",check=False)
    for i,g in enumerate(groups,1):
        status(f"staged commit {i}/{len(groups)}: {g.message}")
        is_bulk = g.id.startswith("bulk-") or g.label.startswith("bulk-")
        if is_bulk and g.file_paths:
            if not stage_files(g.file_paths):
                print(f"Failed on group {i}", file=sys.stderr)
                sys.exit(1)
        else:
            p=build_patch(fhs,hunks,g.hunk_ids)
            if not p.strip():
                continue
            if not stage_patch(p):
                print(f"Failed on group {i}",file=sys.stderr)
                sys.exit(1)
        _commit_group(g)
    verify_tree_or_die(expected_tree, "split verification failed")

def execute_rebase(full_hash,groups,hunks,fhs,expected_tree,dry=False):
    groups = _prune_empty_groups(groups, hunks=hunks)
    status(f"prepare rebase split plan: {len(groups)} commits")
    parent=run_git("rev-parse",f"{full_hash}^").stdout.strip()

    # Write the complete list of files touched by this diff.  Used by the
    # fallback git-add step so it never stages files outside the diff (e.g.
    # the split script itself, editor swap files, etc.).
    all_fps_lst = tempfile.NamedTemporaryFile("w", suffix="_git_split_allfps.lst",
                                              delete=False, dir=tempfile.gettempdir())
    try:
        for fp in sorted(fhs.keys()):
            all_fps_lst.write(fp + "\n")
        all_fps_path = all_fps_lst.name
    finally:
        all_fps_lst.close()

    tmp_files = [all_fps_path]
    plan = []
    for i, g in enumerate(groups, 1):
        is_bulk = g.id.startswith("bulk-") or g.label.startswith("bulk-")
        if is_bulk:
            files = sorted(set(g.file_paths)) if g.file_paths else sorted({hid.split("::", 1)[0] for hid in g.hunk_ids})
            if not files:
                continue
            lf = tempfile.NamedTemporaryFile("w", suffix=f"_git_split_{i}.lst", delete=False, dir=tempfile.gettempdir())
            try:
                for fp in files:
                    lf.write(fp + "\n")
                list_path = lf.name
            finally:
                lf.close()
            tmp_files.append(list_path)
            plan.append(("files", i, g.message, g.body, list_path))
        else:
            patch_text = build_patch(fhs, hunks, g.hunk_ids)
            if not patch_text.strip():
                continue
            pf = tempfile.NamedTemporaryFile("wb", suffix=f"_git_split_{i}.diff", delete=False, dir=tempfile.gettempdir())
            try:
                pf.write(patch_text.encode("utf-8", errors="surrogateescape"))
                patch_path = pf.name
            finally:
                pf.close()
            tmp_files.append(patch_path)
            plan.append(("patch", i, g.message, g.body, patch_path))

    script=["#!/usr/bin/env bash","set -e","git reset HEAD^"]
    for mode, i, msg, body, path in plan:
        mf = tempfile.NamedTemporaryFile("w", suffix=f"_git_split_{i}.msg", delete=False, dir=tempfile.gettempdir())
        try:
            mf.write(msg.strip() + "\n")
            if body:
                mf.write("\n" + body.strip() + "\n")
            msg_path = mf.name
        finally:
            mf.close()
        tmp_files.append(msg_path)
        if mode == "files":
            script += [
                "git reset",
                f'git add -A --pathspec-from-file="{path}"',
                f'git commit -F "{msg_path}"',
            ]
        else:
            script += [
                f'git apply --cached --whitespace=nowarn "{path}"',
                f'git commit -F "{msg_path}"',
            ]
    script += [
        # Catch anything not covered by patches: binary blobs, mode-only changes,
        # new files whose hunks were absent from all groups, etc.
        # Scoped to files in the original diff (--pathspec-from-file) so we never
        # accidentally commit unrelated files that happen to be in the working tree
        # (e.g. the split script itself, editor swap files, IDE artefacts).
        f'_remaining=$(git add -A --dry-run --pathspec-from-file="{all_fps_path}" 2>/dev/null | wc -l)',
        'if [ "$_remaining" -gt 0 ]; then',
        '    echo "git-split: staging remaining files (binary assets / uncovered hunks: $_remaining files)..."',
        f'    git add -A --pathspec-from-file="{all_fps_path}"',
        '    git commit -m "misc: binary assets and remaining files from import"',
        "fi",
        "actual_tree=\"$(git rev-parse HEAD^{tree})\"",
        f'if [ "${{actual_tree}}" != "{expected_tree}" ]; then',
        f'    echo "rebase split verification failed: expected {expected_tree} got ${{actual_tree}}" >&2',
        f'    echo "Files that differ:" >&2',
        f'    git diff-tree --no-commit-id -r --name-status "{expected_tree}" "${{actual_tree}}" | head -40 >&2',
        f'    exit 1',
        f'fi',
        "git rebase --continue",
    ]

    status(f"writing rebase execution script with {len(plan)} steps")
    sp = os.path.join(tempfile.gettempdir(), f"git_split_rebase_{full_hash[:12]}.sh")
    with open(sp, "w") as f:
        f.write("\n".join(script)+"\n")
    os.chmod(sp,0o755)

    print(f"Run:\n  git rebase -i {parent}\n  # change pick {full_hash[:7]} -> edit {full_hash[:7]}\n  bash {sp}")
    if dry:
        return
    if questionary and questionary.confirm("Launch rebase automatically now?",default=False).ask():
        try:
            _auto_rebase(full_hash, parent, sp)
            print("Rebase complete.")
        except subprocess.CalledProcessError as e:
            print(f"Auto rebase failed: {e}", file=sys.stderr)
            print(f"Run manually: bash {sp}")
        finally:
            for tf in tmp_files:
                try:
                    os.unlink(tf)
                except OSError:
                    pass

def _parse_deep_subsystem_roots(raw: Optional[str]) -> dict[str, int]:
    """Parse --deep-subsystem-roots value into a dict.

    Format: 'drivers/foo/drv:4,kernel/bar:3'
    """
    if not raw:
        return {}
    result: dict[str, int] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise click.BadParameter(
                f"Invalid deep subsystem root {entry!r}; expected prefix:depth"
            )
        prefix, _, depth_str = entry.rpartition(":")
        prefix = prefix.strip().strip("/")
        if not prefix:
            raise click.BadParameter(
                f"Invalid deep subsystem root {entry!r}; prefix must be non-empty"
            )
        try:
            depth = int(depth_str.strip())
        except ValueError:
            raise click.BadParameter(
                f"Invalid deep subsystem root {entry!r}; depth must be an integer"
            ) from None
        if depth < 3:
            raise click.BadParameter(
                f"Invalid deep subsystem root {entry!r}; depth must be at least 3"
                " (the default is 2 components, so an override must go deeper)"
            )
        result[prefix] = depth
    return result


def _bundles_to_groups(bundles, hunks):
    """Convert Bundle objects to CommitGroup objects for the existing pipeline."""
    hmap = {h.id: h for h in hunks}
    groups = []
    for bundle in bundles:
        hunk_ids = [h.hunk_id for h in bundle.hunks if h.hunk_id in hmap]
        file_paths = sorted({h.file_path for h in bundle.hunks})
        g = CommitGroup(
            id=bundle.bundle_id,
            label=bundle.preliminary_label,
            message=bundle.preliminary_label,
            hunk_ids=hunk_ids,
            file_paths=file_paths,
        )
        groups.append(g)
    return groups


def enhanced_llm_arbitration(groups, bundles, hunks, api_key, fhs, analysis, body_mode):
    """Refine grouping for low-confidence bundles, then generate commit messages for all groups.

    Stage 1 — merge review (low-confidence only):
      Sends compact bundle summaries (not raw diffs) to Claude and asks which groups
      are clearly about the same topic and should be merged.  Works at the group level
      so it scales to any import size regardless of hunk count.  Batched at
      _MERGE_BATCH_SIZE groups per request.

    Stage 2 — message generation (all groups):
      Calls ai_generate_messages with bundle summary context.
    """
    if not api_key or not anthropic:
        return groups

    bundle_map = {b.bundle_id: b for b in bundles}

    # Build release context preamble
    release_preamble = ""
    if analysis is not None and hasattr(analysis, "release_context") and analysis.release_context is not None:
        rc = analysis.release_context
        v_str = rc.version or "unknown"
        d_str = rc.date or "unknown"
        release_preamble = f"This diff represents vendor release {v_str} dated {d_str}. "

    # --- Stage 1: merge review for low-confidence groups ---
    low_conf = [g for g in groups if bundle_map.get(g.id) and bundle_map[g.id].confidence == "low"]
    high_conf = [g for g in groups if g not in low_conf]

    if low_conf and api_key:
        status(f"merge review: {len(low_conf)} low-confidence groups across {(len(low_conf) + _MERGE_BATCH_SIZE - 1) // _MERGE_BATCH_SIZE} batch(es)")
        try:
            merged_low = []
            for i in range(0, len(low_conf), _MERGE_BATCH_SIZE):
                batch = low_conf[i:i + _MERGE_BATCH_SIZE]
                batch_label = f"batch {i // _MERGE_BATCH_SIZE + 1}" if len(low_conf) > _MERGE_BATCH_SIZE else ""
                if batch_label:
                    status(f"merge review: {batch_label} ({len(batch)} groups)...")
                merged_low.extend(_ai_merge_groups(batch, bundle_map, api_key, release_preamble))
            groups = high_conf + merged_low
        except Exception as e:
            status(f"merge review failed ({e}), proceeding with original grouping")

    # --- Stage 2: message generation for all groups ---
    summaries = [bundle_map[g.id].summary() for g in groups if g.id in bundle_map]
    context = release_preamble
    if summaries:
        context += "Bundle analysis:\n" + "\n\n".join(summaries[:10])

    try:
        ai_generate_messages(
            groups, hunks, api_key,
            original=context or None,
            body_mode=body_mode,
            fhs=fhs,
        )
    except Exception as e:
        status(f"message generation failed ({e}), falling back")
        try:
            ai_generate_messages(groups, hunks, api_key, original=release_preamble or None, body_mode=body_mode, fhs=fhs)
        except Exception as e2:
            status(f"fallback message generation also failed: {e2}")

    return groups


@click.group()
@click.option("--api-key", envvar="ANTHROPIC_API_KEY", help="Anthropic API key")
@click.option("--voyage-api-key", envvar="VOYAGE_API_KEY", default=None, help="Voyage AI API key (optional, enables embedding refinement)")
@click.pass_context
def cli(ctx,api_key,voyage_api_key):
    load_env_defaults()
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not voyage_api_key:
        voyage_api_key = os.environ.get("VOYAGE_API_KEY")
    ctx.ensure_object(dict)
    ctx.obj["api_key"] = api_key
    ctx.obj["voyage_api_key"] = voyage_api_key

@cli.command()
@click.option("--staged/--unstaged", default=True)
@click.option("--dry-run", is_flag=True)
@click.option("--message-body", type=click.Choice(["off","auto","always"]), default="auto", show_default=True)
@click.option("--kernel-root", default=None, help="Manual kernel root path override")
@click.option("--no-release-context", is_flag=True, default=False, help="Skip release context extraction from HEAD commit")
@click.option("--export-bundles", is_flag=True, default=False, help="Exit after bundling and print bundle report")
@click.option("--export-format", type=click.Choice(["md","json"]), default="md", show_default=True, help="Format for --export-bundles")
@click.option("--export-file", default=None, help="Write export report to this file instead of stdout")
@click.option("--attachment-threshold", default=4.0, show_default=True, type=float, help="Minimum affinity score for hunk attachment")
@click.option("--max-anchor-breadth", default=MAX_ANCHOR_BREADTH, show_default=True, type=int, help="Maximum new-file count per anchor bundle before splitting")
@click.option("--skip-components", default=None, help="Comma-separated list of components to skip")
@click.option("--deep-subsystem-roots", default=None, help="Extra deep subsystem roots, format: 'drivers/foo/drv:4,kernel/bar:3'")
@click.pass_context
def split(ctx,staged,dry_run,message_body,kernel_root,no_release_context,export_bundles,export_format,export_file,attachment_threshold,max_anchor_breadth,skip_components,deep_subsystem_roots):
    assert_repo(); assert_clean_state()
    api=ctx.obj.get("api_key"); voyage_api=ctx.obj.get("voyage_api_key")
    diff_mode = "staged" if staged else "unstaged"
    print(f"git-split: splitting {diff_mode} changes")
    status("reading git diff...")
    raw=get_diff(staged=staged)
    if not raw.strip():
        print("No changes found.")
        return
    status("parsing diff into hunks...")
    fhs,base_hunks=parse_diff(raw)

    all_file_paths = list(fhs.keys())
    analysis = analyze_repo(all_file_paths, repo_path=".",
                            kernel_root_override=kernel_root,
                            no_release_context=no_release_context)
    status(f"repo mode: {analysis.mode.value}, kernel_root: {analysis.kernel_root!r}")

    skip_set = set()
    if skip_components:
        skip_set = {c.strip() for c in skip_components.split(",") if c.strip()}

    extra_deep_roots = _parse_deep_subsystem_roots(deep_subsystem_roots)
    partition_result = partition(base_hunks, fhs, analysis, deep_roots=extra_deep_roots)
    if skip_set:
        partition_result.hunks = [h for h in partition_result.hunks if h.component not in skip_set]

    features = extract_features(partition_result.hunks)
    bundles = build_bundles(features, partition_result,
                            attachment_threshold=attachment_threshold,
                            max_anchor_breadth=max_anchor_breadth)
    bundles = refine_bundles(bundles, voyage_api)

    if export_bundles:
        report = generate_export_report(bundles, analysis, export_format=export_format)
        if export_file:
            Path(export_file).write_text(report, encoding="utf-8")
            print(f"Export written to {export_file}")
        else:
            print(report)
        return

    groups = _bundles_to_groups(bundles, base_hunks)
    groups = enhanced_llm_arbitration(groups, bundles, base_hunks, api, fhs=fhs,
                                       analysis=analysis, body_mode=message_body)
    groups = _prune_empty_groups(groups, hunks=base_hunks)
    status("entering interactive review...")
    groups = interactive_edit(groups, base_hunks, fhs=fhs, api_key=api)
    groups = _prune_empty_groups(groups, hunks=base_hunks)
    expected_tree = run_git("write-tree").stdout.strip() if staged else expected_tree_from_patch(raw, base_ref="HEAD")
    execute_staged(groups, base_hunks, fhs, expected_tree=expected_tree, dry=dry_run)

@cli.command()
@click.argument("commit_hash", required=False)
@click.option("--dry-run", is_flag=True)
@click.option("--message-body", type=click.Choice(["off","auto","always"]), default="auto", show_default=True)
@click.option("--kernel-root", default=None, help="Manual kernel root path override")
@click.option("--no-release-context", is_flag=True, default=False, help="Skip release context extraction from HEAD commit")
@click.option("--skip-components", default=None, help="Comma-separated list of components to skip")
@click.option("--attachment-threshold", default=4.0, show_default=True, type=float, help="Minimum affinity score for hunk attachment")
@click.option("--max-anchor-breadth", default=MAX_ANCHOR_BREADTH, show_default=True, type=int, help="Maximum new-file count per anchor bundle before splitting")
@click.option("--deep-subsystem-roots", default=None, help="Extra deep subsystem roots, format: 'drivers/foo/drv:4,kernel/bar:3'")
@click.pass_context
def rebase(ctx,commit_hash,dry_run,message_body,kernel_root,no_release_context,skip_components,attachment_threshold,max_anchor_breadth,deep_subsystem_roots):
    assert_repo(); assert_clean_state()
    api=ctx.obj.get("api_key"); voyage_api=ctx.obj.get("voyage_api_key")
    if not commit_hash: commit_hash="HEAD"
    full=run_git("rev-parse",commit_hash).stdout.strip()
    # Ensure the commit has a parent (can't split a root commit)
    try:
        run_git("rev-parse","--verify",f"{full}^")
    except subprocess.CalledProcessError:
        print(f"Error: commit {full[:12]} has no parent — cannot split a root commit.", file=sys.stderr)
        sys.exit(1)
    original = get_commit_msg(full)
    print(f"Commit: {full[:12]} {original[:70]}")
    print("Parsing commit diff...")
    status("reading commit diff...")
    raw=get_commit_diff(full)
    if not raw.strip():
        print("No diff found for this commit.")
        return
    status("parsing commit diff into hunks...")
    fhs,base_hunks=parse_diff(raw)
    status("captured expected commit tree for no-loss verification...")
    parent_ref = run_git("rev-parse", f"{full}^").stdout.strip()
    expected_tree = run_git("rev-parse", f"{full}^{{tree}}").stdout.strip()
    preflight_patch_check(raw, parent_ref)

    all_file_paths = list(fhs.keys())
    analysis = analyze_repo(all_file_paths, repo_path=".",
                            kernel_root_override=kernel_root,
                            no_release_context=no_release_context)
    status(f"repo mode: {analysis.mode.value}, kernel_root: {analysis.kernel_root!r}")

    skip_set = set()
    if skip_components:
        skip_set = {c.strip() for c in skip_components.split(",") if c.strip()}

    extra_deep_roots = _parse_deep_subsystem_roots(deep_subsystem_roots)
    partition_result = partition(base_hunks, fhs, analysis, deep_roots=extra_deep_roots)
    if skip_set:
        partition_result.hunks = [h for h in partition_result.hunks if h.component not in skip_set]

    features = extract_features(partition_result.hunks)
    bundles = build_bundles(features, partition_result,
                            attachment_threshold=attachment_threshold,
                            max_anchor_breadth=max_anchor_breadth)
    bundles = refine_bundles(bundles, voyage_api)

    groups = _bundles_to_groups(bundles, base_hunks)
    groups = enhanced_llm_arbitration(groups, bundles, base_hunks, api, fhs=fhs,
                                       analysis=analysis, body_mode=message_body)
    groups = _prune_empty_groups(groups, hunks=base_hunks)
    status("entering interactive review...")
    groups = interactive_edit(groups, base_hunks, fhs=fhs, api_key=api, original_message=original)
    groups = _prune_empty_groups(groups, hunks=base_hunks)
    execute_rebase(full, groups, base_hunks, fhs, expected_tree=expected_tree, dry=dry_run)

@cli.command()
def check():
    try: print(run_git("--version").stdout.strip())
    except Exception: print("git not found")
    print("ANTHROPIC_API_KEY set" if os.environ.get("ANTHROPIC_API_KEY") else "ANTHROPIC_API_KEY not set")
    print("VOYAGE_API_KEY set" if os.environ.get("VOYAGE_API_KEY") else "VOYAGE_API_KEY not set (embedding refinement disabled)")
    print("structural engine: ok")

if __name__ == "__main__":
    cli(obj={})

