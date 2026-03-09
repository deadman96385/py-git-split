#!/usr/bin/env python3
import os, sys, re, json, subprocess, tempfile, warnings
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import click

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


def expected_tree_from_patch(raw_patch, base_ref="HEAD"):
    status(f"computing expected tree from patch (base={base_ref})")
    if not raw_patch.strip():
        return run_git("rev-parse", f"{base_ref}^{{tree}}" if base_ref != "HEAD" else "HEAD^{tree}").stdout.strip()
    pf = tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False, dir=tempfile.gettempdir())
    idxf = tempfile.NamedTemporaryFile("w", suffix=".idx", delete=False, dir=tempfile.gettempdir())
    try:
        pf.write(raw_patch)
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
        run_git_env("apply", "--cached", "--whitespace=nowarn", patch_path, env=env)
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

def get_diff(staged=True):
    if staged:
        return run_git("diff", "--cached", "--unified=3").stdout
    return run_git("diff", "--unified=3").stdout

def get_commit_diff(h): return run_git("show","--unified=3",h).stdout

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
    for line in raw.splitlines():
        m = fr.match(line)
        if m:
            flush_hunk(); flush_fh(); fp=m.group(2); fhl=[line]; hh=None; hl=[]; continue
        if fp and any(line.startswith(p) for p in ("index ","--- ","+++ ","new file","deleted file","Binary","old mode","new mode","rename ","similarity ")):
            if hh is None: fhl.append(line)
            continue
        if hr.match(line):
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
    with tempfile.NamedTemporaryFile("w",suffix=".patch",delete=False) as f: f.write(p); n=f.name
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
def _scope(fp):
    p=fp.split('/')
    return f"{p[0]}-{p[1]}" if len(p)>=2 else (p[0] if p else "repo")

def summarize(hunks,fhs):
    a=sum(h.added for h in hunks); r=sum(h.removed for h in hunks); f=len(fhs); n=sum(1 for k in fhs if _is_new(fhs.get(k))); t=a+r
    return {"hunks":len(hunks),"files":f,"added":a,"removed":r,"new_files":n,"add_ratio":(a/t if t else 1.0),"new_file_ratio":(n/f if f else 0.0)}

def large_add_mode(s):
    return (s["files"]>=2500 or s["hunks"]>=5000 or s["added"]>=800000) and (s["add_ratio"]>=0.94 or s["new_file_ratio"]>=0.70)

def _msg_type(msg, default="chore"):
    m=re.match(r'^([a-z]+)(?:\(|:)', (msg or '').strip())
    return m.group(1) if m else default

def _group_message(groups, hunks, fhs):
    hmap={h.id:h for h in hunks}
    for g in groups:
        hs=[hmap[hid] for hid in g.hunk_ids if hid in hmap]
        if not hs:
            continue
        files=sorted({h.filepath for h in hs})
        if not files:
            continue
        added=sum(h.added for h in hs)
        removed=sum(h.removed for h in hs)
        new_files=sum(1 for fp in files if _is_new(fhs.get(fp)))

        by_scope=defaultdict(int)
        by_area=defaultdict(int)
        for fp in files:
            by_scope[_scope(fp)] += 1
            p=fp.split('/')
            area=p[2] if len(p)>=3 else p[-1]
            by_area[area] += 1

        primary_scope=sorted(by_scope.items(), key=lambda x: (-x[1], x[0]))[0][0]
        top_areas=[k for k,_ in sorted(by_area.items(), key=lambda x: (-x[1], x[0]))[:2]]

        if len(files)==new_files and new_files>0:
            verb="import"
        elif removed==0 and added>0:
            verb="add"
        elif added==0 and removed>0:
            verb="remove"
        else:
            verb="update"

        ctype=_msg_type(g.message, default=("feat" if verb in ("import", "add") else "chore"))
        msg=f"{ctype}({primary_scope}): {verb} {len(files)} files (+{added}/-{removed})"
        if top_areas:
            msg += f" [{', '.join(top_areas)}]"

        if len(msg) > 110:
            msg=f"{ctype}({primary_scope}): {verb} {len(files)} files (+{added}/-{removed})"

        g.message=msg
    return groups

def bulk_groups(hunks,fhs,max_files=300,max_hunks=2000):
    byf=defaultdict(list)
    for h in hunks: byf[h.filepath].append(h.id)
    for fp in fhs.keys():
        byf.setdefault(fp, [])
    bys=defaultdict(list)
    for fp in sorted(byf): bys[_scope(fp)].append(fp)
    groups=[]; gid=0

    def add_group(sc, files, batch_no):
        nonlocal gid
        hids=[]
        for fp in files:
            hids += byf[fp]
        nf=sum(1 for fp in files if _is_new(fhs.get(fp)))
        typ="feat" if nf==len(files) else "chore"
        groups.append(CommitGroup(f"bulk-{gid}",f"bulk-{sc}-{batch_no}",f"{typ}({sc}): import batch {batch_no}",hids,file_paths=list(files)))
        gid+=1

    for sc in sorted(bys):
        files=bys[sc]
        scope_hunks=sum(len(byf[fp]) for fp in files)
        scope_is_large=(len(files)>=max_files or scope_hunks>max_hunks)
        force_scope_commit=(
            sc.startswith("kernel-")
            or sc.startswith("vendor-")
            or sc.startswith("prebuilts-")
            or sc.startswith("bootable-bootloader-")
            or scope_is_large
        )

        # Keep large scopes and kernel/* scopes as dedicated commits.
        if force_scope_commit:
            add_group(sc, files, 1)
            continue

        chunk=[]; ch=0; bn=1
        for fp in files:
            fh=len(byf[fp])
            if chunk and (len(chunk)>=max_files or ch+fh>max_hunks):
                add_group(sc, chunk, bn)
                bn += 1
                chunk=[]; ch=0
            chunk.append(fp); ch += fh
        if chunk:
            add_group(sc, chunk, bn)
    return groups

def _require_anthropic(api_key):
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for semantic split modes")
    if not anthropic:
        raise RuntimeError("anthropic package is required for semantic split modes")


def _require_treesitter():
    try:
        from tree_sitter_languages import get_parser  # type: ignore
        return get_parser
    except Exception as e:
        raise RuntimeError(f"tree-sitter is required (install tree_sitter_languages): {e}")


def _get_ts_parser(lang, get_parser):
    try:
        return get_parser(lang)
    except TypeError as e:
        raise RuntimeError(
            "tree-sitter parser initialization failed for "
            f"language '{lang}': {e}. "
            "This usually means incompatible versions of tree_sitter and tree_sitter_languages are installed."
        )
    except Exception as e:
        raise RuntimeError(f"failed to initialize tree-sitter parser for language '{lang}': {e}")


def _ext_to_lang(fp):
    ext = Path(fp).suffix.lower()
    m = {
        ".c":"c", ".h":"c", ".cc":"cpp", ".cpp":"cpp", ".cxx":"cpp", ".hpp":"cpp",
        ".java":"java", ".kt":"kotlin", ".kts":"kotlin",
        ".js":"javascript", ".jsx":"javascript", ".ts":"typescript", ".tsx":"tsx",
        ".py":"python", ".rs":"rust", ".go":"go", ".rb":"ruby", ".php":"php",
        ".swift":"swift", ".m":"objc", ".mm":"cpp", ".cs":"c_sharp",
    }
    return m.get(ext)


def _extract_hunk_new_start(header):
    m = re.search(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", header)
    if not m:
        warnings.warn(f"Could not parse hunk header: {header!r}")
        return 1
    return int(m.group(1))


def _changed_blocks(lines):
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        while i < n and not (lines[i].startswith('+') or lines[i].startswith('-')):
            i += 1
        if i >= n:
            break
        j = i
        while j < n and (lines[j].startswith('+') or lines[j].startswith('-')):
            j += 1
        blocks.append((i, j))
        i = j
    return blocks


def _unit_lines_with_context(lines, start, end, ctx=3):
    a = max(0, start-ctx)
    b = min(len(lines), end+ctx)
    return lines[a:b]


def _line_unit_lines(lines, pos, ctx=2):
    out=[]
    a=max(0, pos-ctx)
    b=min(len(lines), pos+ctx+1)
    for i in range(a, b):
        ln=lines[i]
        if i==pos:
            out.append(ln)
        elif ln.startswith(' '):
            out.append(ln)
    return out


def _read_file_for_context(fp, ref=None):
    if ref:
        try:
            return run_git("show", f"{ref}:{fp}").stdout
        except Exception:
            pass
    try:
        if Path(fp).exists():
            return Path(fp).read_text(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        return run_git("show", f"HEAD:{fp}").stdout
    except Exception:
        return ""


def _collect_symbol_nodes(root):
    out=[]
    stack=[root]
    while stack:
        n=stack.pop()
        t=str(getattr(n, "type", ""))
        if any(k in t for k in ("function", "method", "class")):
            out.append((n.start_point[0]+1, n.end_point[0]+1, t))
        try:
            stack.extend(reversed(list(n.children)))
        except Exception:
            pass
    out.sort(key=lambda x:(x[0], x[1]))
    return out


def _symbol_for_line(fp, line_no, get_parser, cache, line_cache, ref=None):
    key=(fp, line_no)
    if key in line_cache:
        return line_cache[key]

    if fp not in cache:
        lang=_ext_to_lang(fp)
        if not lang:
            cache[fp]=[]
        else:
            parser=_get_ts_parser(lang, get_parser)
            src=_read_file_for_context(fp, ref=ref).encode("utf-8", errors="replace")
            tree=parser.parse(src)
            cache[fp]=_collect_symbol_nodes(tree.root_node)
    syms=cache.get(fp,[])
    matches=[(a,b,name) for a,b,name in syms if a <= line_no <= b]
    if matches:
        best=min(matches, key=lambda x: x[1]-x[0])
        line_cache[key]=best[2]
        return best[2]
    line_cache[key]="file"
    return "file"


def _build_new_line_map(hunk):
    new_ln=_extract_hunk_new_start(hunk.header)
    out=[0]*len(hunk.lines)
    for i,ln in enumerate(hunk.lines):
        if ln.startswith('+'):
            out[i]=new_ln
            new_ln += 1
        elif ln.startswith(' '):
            out[i]=new_ln
            new_ln += 1
    return out

def split_hunks_context(hunks, granularity="adaptive", line_safety="balanced", ref=None):
    if granularity not in ("adaptive", "fine"):
        raise RuntimeError("granularity must be adaptive or fine")
    get_parser = _require_treesitter()
    symbol_cache={}
    line_symbol_cache={}

    total = len(hunks)
    status(f"context split start: {total} hunks, granularity={granularity}, line_safety={line_safety}")
    out = []
    for i, h in enumerate(hunks, 1):
        blocks = _changed_blocks(h.lines)
        if not blocks:
            out.append(h)
            continue

        new_line_map = _build_new_line_map(h)

        if i % 250 == 0 or i == total:
            status(f"context split progress: {i}/{total} hunks -> {len(out)} units")
        for bi, (b0, b1) in enumerate(blocks):
            plus_pos=[k for k in range(b0, b1) if h.lines[k].startswith('+')]
            has_minus=any(h.lines[k].startswith('-') for k in range(b0, b1))
            add_only=bool(plus_pos) and not has_minus

            do_line_split = (granularity == "fine" and add_only)
            if granularity == "adaptive" and add_only and len(plus_pos) > 1:
                symbols={
                    _symbol_for_line(h.filepath, new_line_map[pos], get_parser, symbol_cache, line_symbol_cache, ref=ref)
                    for pos in plus_pos
                    if new_line_map[pos] > 0
                }
                do_line_split = len(symbols) > 1

            if do_line_split:
                ctx = 2 if line_safety=="balanced" else (1 if line_safety=="conservative" else 3)
                for li, pos in enumerate(plus_pos):
                    ul = _line_unit_lines(h.lines, pos, ctx=ctx)
                    out.append(Hunk(f"{h.id}::b{bi}::l{li}", h.filepath, h.header, ul, h.index))
            else:
                ul = _unit_lines_with_context(h.lines, b0, b1, ctx=3)
                out.append(Hunk(f"{h.id}::b{bi}", h.filepath, h.header, ul, h.index))

    out.sort(key=lambda x: (x.filepath, x.index, x.id))
    status(f"context split done: {len(out)} units")
    return out


def _group_stats(g, hunks, fhs):
    hmap={h.id:h for h in hunks}
    hs=[hmap[hid] for hid in g.hunk_ids if hid in hmap]
    files=sorted(set({h.filepath for h in hs}) | set(g.file_paths or []))
    added=sum(h.added for h in hs)
    removed=sum(h.removed for h in hs)
    new_files=sum(1 for fp in files if _is_new(fhs.get(fp)))
    add_ratio=(added/(added+removed) if (added+removed) else 1.0)
    new_ratio=(new_files/len(files) if files else 0.0)
    scopes=defaultdict(int)
    for fp in files:
        scopes[_scope(fp)] += 1
    scope=sorted(scopes.items(), key=lambda x:(-x[1], x[0]))[0][0] if scopes else "repo"
    return {
        "files": len(files),
        "new_files": new_files,
        "added": added,
        "removed": removed,
        "add_ratio": add_ratio,
        "new_ratio": new_ratio,
        "scope": scope,
    }


def _human_scope(scope):
    parts=scope.split('-') if scope else ["repo"]
    if len(parts) >= 2 and parts[0] == "kernel":
        return f"{' '.join(parts[1:])} kernel"
    return " ".join(parts)


def _is_bulk_import_group(stats):
    huge = stats["files"] >= 200 or stats["added"] >= 50000
    import_like = stats["new_ratio"] >= 0.90 and stats["add_ratio"] >= 0.98 and stats["removed"] <= 200
    return huge and import_like


def _set_bulk_import_message(g, stats):
    target=_human_scope(stats["scope"])
    g.message=f"feat: import {target}"
    g.body=None


def apply_hybrid_bulk_messages(groups, hunks, fhs, api_key, original=None, body_mode="auto"):
    ai_groups_only=[]
    bulk_count=0
    for g in groups:
        st=_group_stats(g, hunks, fhs)
        if _is_bulk_import_group(st):
            _set_bulk_import_message(g, st)
            bulk_count += 1
        else:
            ai_groups_only.append(g)

    status(f"bulk message routing: deterministic_bulk={bulk_count}, ai_groups={len(ai_groups_only)}")
    if ai_groups_only:
        ai_generate_messages(ai_groups_only, hunks, api_key, original=original, body_mode=body_mode, fhs=fhs)
    return groups


def enforce_commit_cap(groups, hunks, max_commits=25):
    status(f"enforcing commit cap: current={len(groups)} cap={max_commits}")
    if max_commits <= 0:
        raise RuntimeError("max-generated-commits must be > 0")
    groups = [g for g in groups if g.hunk_ids or g.file_paths]
    if len(groups) <= max_commits:
        status("commit cap not needed")
        return groups

    hmap = {h.id: h for h in hunks}
    def group_scope(g):
        files = list(g.file_paths or [])
        if not files:
            files = [hmap[hid].filepath for hid in g.hunk_ids if hid in hmap]
        if not files:
            return "misc"
        c = defaultdict(int)
        for fp in files:
            c[_scope(fp)] += 1
        return sorted(c.items(), key=lambda x:(-x[1], x[0]))[0][0]

    # merge smallest groups first into nearest scope match, deterministic
    while len(groups) > max_commits:
        groups = sorted(groups, key=lambda g: (len(g.hunk_ids) + len(g.file_paths or []), g.id))
        g = groups[0]
        if len(groups) < 2:
            raise RuntimeError("cannot reconcile commit cap")
        src_scope = group_scope(g)
        target_idx = None
        for i in range(1, len(groups)):
            if group_scope(groups[i]) == src_scope:
                target_idx = i
                break
        if target_idx is None:
            target_idx = 1
        groups[target_idx].hunk_ids.extend(g.hunk_ids)
        if g.file_paths:
            groups[target_idx].file_paths = sorted(set(groups[target_idx].file_paths + g.file_paths))
        groups = groups[1:]
    return groups


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
        scopes[_scope(fp)] += 1
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


def ai_generate_messages(groups, hunks, api_key, original=None, body_mode="auto", fhs=None):
    status(f"AI message generation start: {len(groups)} groups (body_mode={body_mode})")
    _require_anthropic(api_key)
    c=anthropic.Anthropic(api_key=api_key)
    hmap={h.id:h for h in hunks}
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
        payload.append({
            "id": g.id,
            "files": files[:80],
            "file_count": len(files),
            "new_files": new_files,
            "file_only_count": file_only_count,
            "hunks": len(hs),
            "added": added,
            "removed": removed,
            "preview": [{"file":h.filepath,"header":h.header,"sample":"\n".join(h.lines[:6])} for h in hs[:8]],
        })
    req={
        "original": original or "",
        "body_mode": body_mode,
        "commits": payload,
        "format": "Return JSON only: {commits:[{id,subject,body_lines:[...]}]} subject <= 100 chars, format strictly as type: scope: summary (example: fix: kernel: remove unused keys). Never say no file changes when file_count > 0.",
    }
    status("waiting for AI message response...")
    r=c.messages.create(model="claude-sonnet-4-20250514",max_tokens=8192,system="Return JSON only.",messages=[{"role":"user","content":json.dumps(req)}])
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
        raise RuntimeError(f"AI returned invalid JSON for message generation: {e}") from e
    by={x.get("id"):x for x in d.get("commits",[])}
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

def ai_groups(hunks,api_key,msg=None):
    status(f"AI grouping start: {len(hunks)} units")
    _require_anthropic(api_key)
    c=anthropic.Anthropic(api_key=api_key)
    arr=[{"id":h.id,"file":h.filepath,"hunk_header":h.header,"preview":"\n".join(h.lines[:12]),"added":h.added,"removed":h.removed} for h in hunks]
    req={"original": msg or "", "commits": arr, "format": "Return JSON only: {commits:[{label,message,hunk_ids,confidence}]}"}
    r=c.messages.create(model="claude-sonnet-4-20250514",max_tokens=8192,system="Return JSON only.",messages=[{"role":"user","content":json.dumps(req)}])
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

    tmp_files = []
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
            pf = tempfile.NamedTemporaryFile("w", suffix=f"_git_split_{i}.diff", delete=False, dir=tempfile.gettempdir())
            try:
                pf.write(patch_text)
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
        "actual_tree=\"$(git rev-parse HEAD^{tree})\"",
        f'test "${{actual_tree}}" = "{expected_tree}" || (echo "rebase split verification failed: expected {expected_tree} got ${{actual_tree}}" >&2; exit 1)',
        "git rebase --continue",
    ]

    status(f"writing rebase execution script with {len(plan)} steps")
    with tempfile.NamedTemporaryFile("w",suffix=".sh",delete=False,dir=tempfile.gettempdir()) as f:
        f.write("\n".join(script)+"\n")
        sp=f.name
    tmp_files.append(sp)
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

@click.group()
@click.option("--api-key", envvar="ANTHROPIC_API_KEY", help="Anthropic API key")
@click.pass_context
def cli(ctx,api_key):
    load_env_defaults()
    warnings.filterwarnings(
        "ignore",
        message=r"Language\(path, name\) is deprecated\. Use Language\(ptr, name\) instead\.",
        category=FutureWarning,
    )
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    ctx.ensure_object(dict); ctx.obj["api_key"]=api_key

@cli.command()
@click.option("--staged/--unstaged", default=True)
@click.option("--dry-run", is_flag=True)
@click.option("--max-files-per-group", default=300, show_default=True, type=int)
@click.option("--max-hunks-per-group", default=2000, show_default=True, type=int)
@click.option("--force-bulk", is_flag=True)
@click.option("--granularity", type=click.Choice(["adaptive","fine"]), default="adaptive", show_default=True)
@click.option("--line-safety", type=click.Choice(["conservative","balanced","aggressive"]), default="balanced", show_default=True)
@click.option("--max-generated-commits", default=25, show_default=True, type=int)
@click.option("--message-body", type=click.Choice(["off","auto","always"]), default="auto", show_default=True)
@click.pass_context
def split(ctx,staged,dry_run,max_files_per_group,max_hunks_per_group,force_bulk,granularity,line_safety,max_generated_commits,message_body):
    assert_repo(); api=ctx.obj.get("api_key")
    mode = "staged" if staged else "unstaged"
    print(f"git-split: splitting {mode} changes")
    status("reading git diff...")
    raw=get_diff(staged=staged)
    if not raw.strip():
        print("No changes found.")
        return
    status("parsing diff into hunks...")
    fhs,base_hunks=parse_diff(raw)
    base_summary=summarize(base_hunks,fhs)
    use_bulk = force_bulk or large_add_mode(base_summary)

    if use_bulk:
        hunks = base_hunks
        s = base_summary
        print(f"Found {len(hunks)} change units across {len(fhs)} files")
        if len(hunks) <= 1:
            print("Only one unit; nothing to split.")
            return
        reason = "forced by --force-bulk" if force_bulk else "auto-selected for very large addition-heavy diff"
        print(f"Using bulk mode: {reason}")
        print(f"Stats: files={int(s['files'])}, hunks={int(s['hunks'])}, +{int(s['added'])}/-{int(s['removed'])}, new_files={int(s['new_files'])}, add_ratio={s['add_ratio']:.2f}")
        groups = bulk_groups(hunks,fhs,max_files_per_group,max_hunks_per_group)
    else:
        hunks=split_hunks_context(base_hunks,granularity=granularity,line_safety=line_safety)
        s=summarize(hunks,fhs)
        print(f"Found {len(hunks)} change units across {len(fhs)} files")
        if len(hunks) <= 1:
            print("Only one unit; nothing to split.")
            return
        groups = ai_groups(hunks,api)

    status("building final commit groups...")
    groups = enforce_commit_cap(groups, hunks, max_commits=max_generated_commits)
    if use_bulk:
        groups = apply_hybrid_bulk_messages(groups, hunks, fhs, api, body_mode=message_body)
    else:
        groups = ai_generate_messages(groups, hunks, api, body_mode=message_body, fhs=fhs)
    groups = _prune_empty_groups(groups, hunks=hunks)
    status("entering interactive review...")
    groups = interactive_edit(groups, hunks, fhs=fhs, api_key=api)
    groups = _prune_empty_groups(groups, hunks=hunks)
    status("capturing expected final tree for no-loss verification...")
    expected_tree = run_git("write-tree").stdout.strip() if staged else expected_tree_from_patch(raw, base_ref="HEAD")
    execute_staged(groups,hunks,fhs,expected_tree=expected_tree,dry=dry_run)

@cli.command()
@click.argument("commit_hash", required=False)
@click.option("--dry-run", is_flag=True)
@click.option("--max-files-per-group", default=300, show_default=True, type=int)
@click.option("--max-hunks-per-group", default=2000, show_default=True, type=int)
@click.option("--force-bulk", is_flag=True)
@click.option("--granularity", type=click.Choice(["adaptive","fine"]), default="adaptive", show_default=True)
@click.option("--line-safety", type=click.Choice(["conservative","balanced","aggressive"]), default="balanced", show_default=True)
@click.option("--max-generated-commits", default=25, show_default=True, type=int)
@click.option("--message-body", type=click.Choice(["off","auto","always"]), default="auto", show_default=True)
@click.pass_context
def rebase(ctx,commit_hash,dry_run,max_files_per_group,max_hunks_per_group,force_bulk,granularity,line_safety,max_generated_commits,message_body):
    assert_repo(); api=ctx.obj.get("api_key")
    if not commit_hash: commit_hash="HEAD"
    full=run_git("rev-parse",commit_hash).stdout.strip()
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
    base_summary=summarize(base_hunks,fhs)
    use_bulk = force_bulk or large_add_mode(base_summary)

    if use_bulk:
        hunks = base_hunks
        s = base_summary
        print(f"Found {len(hunks)} change units across {len(fhs)} files")
        if len(hunks) <= 1:
            print("Only one unit; nothing to split.")
            return
        reason = "forced by --force-bulk" if force_bulk else "auto-selected for very large addition-heavy diff"
        print(f"Using bulk mode: {reason}")
        print(f"Stats: files={int(s['files'])}, hunks={int(s['hunks'])}, +{int(s['added'])}/-{int(s['removed'])}, new_files={int(s['new_files'])}, add_ratio={s['add_ratio']:.2f}")
        groups = bulk_groups(hunks,fhs,max_files_per_group,max_hunks_per_group)
    else:
        hunks=split_hunks_context(base_hunks,granularity=granularity,line_safety=line_safety,ref=full)
        s=summarize(hunks,fhs)
        print(f"Found {len(hunks)} change units across {len(fhs)} files")
        if len(hunks) <= 1:
            print("Only one unit; nothing to split.")
            return
        groups = ai_groups(hunks,api,original)

    groups = enforce_commit_cap(groups, hunks, max_commits=max_generated_commits)
    if use_bulk:
        groups = apply_hybrid_bulk_messages(groups, hunks, fhs, api, original=original, body_mode=message_body)
    else:
        groups = ai_generate_messages(groups, hunks, api, original=original, body_mode=message_body, fhs=fhs)
    groups = _prune_empty_groups(groups, hunks=hunks)
    status("entering interactive review...")
    groups = interactive_edit(groups, hunks, fhs=fhs, api_key=api, original_message=original)
    groups = _prune_empty_groups(groups, hunks=hunks)
    status("capturing expected commit tree for no-loss verification...")
    expected_tree = run_git("rev-parse", f"{full}^{{tree}}").stdout.strip()
    execute_rebase(full,groups,hunks,fhs,expected_tree=expected_tree,dry=dry_run)

@cli.command()
def check():
    try: print(run_git("--version").stdout.strip())
    except Exception: print("git not found")
    print("ANTHROPIC_API_KEY set" if os.environ.get("ANTHROPIC_API_KEY") else "ANTHROPIC_API_KEY not set")
    try:
        gp = _require_treesitter()
        _get_ts_parser("c", gp)
        print("tree-sitter parser init: ok")
    except Exception as e:
        print(f"tree-sitter parser init: failed: {e}")

if __name__ == "__main__":
    cli(obj={})

