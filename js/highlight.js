/* ============================================================
   highlight.js — tiny dependency-free syntax highlighter
   Supports: js, ts, py, go, rust, php, bash, batch, json, sql, cypher, graphql, env, http
   ============================================================ */
(function () {
  "use strict";

  const KEYWORDS = {
    common: ["if","else","for","while","return","break","continue","in","of","new","try","catch","finally","throw","switch","case","default","do","class","extends","import","export","from","as","async","await","yield","this","super","null","true","false","void","typeof","instanceof"],
    ts: ["const","let","var","function","interface","type","enum","implements","public","private","protected","readonly","static","abstract","declare","namespace","keyof","infer","satisfies","get","set","constructor"],
    js: ["const","let","var","function","get","set","constructor"],
    py: ["def","lambda","pass","None","True","False","and","or","not","is","elif","with","global","nonlocal","assert","del","raise","except","finally","import","from","as","class","async","await","return","yield","match"],
    go: ["func","package","var","const","type","struct","interface","map","chan","go","defer","select","range","fallthrough","goto","nil","make","append","len","cap","string","int","int64","bool","byte","rune","error","float64","uint"],
    rust: ["fn","let","mut","const","struct","enum","impl","trait","pub","use","mod","match","move","ref","dyn","where","unsafe","async","await","self","Self","crate","super","loop","Some","None","Ok","Err","Option","Result","Box","Vec","String","str","usize","i32","u32","i64","u64","bool"],
    php: ["function","public","private","protected","static","const","namespace","use","class","interface","trait","extends","implements","abstract","final","new","echo","print","fn","match","readonly","enum","array","null","true","false","self","parent","require","include"],
    bash: ["echo","cd","export","source","sudo","then","fi","do","done","in","function","local","if","else","elif","for","while","until","case","esac","select","time","return","exit","set","unset","shift","read","printf","test","declare","readonly","eval","exec","trap","alias","unalias","getopts","break","continue","then"],
    batch: ["echo","set","if","else","for","in","do","goto","call","exit","rem","setlocal","endlocal","enabledelayedexpansion","enableextensions","disabledelayedexpansion","pause","cls","title","color","cd","chdir","dir","copy","xcopy","robocopy","move","del","erase","ren","rename","md","mkdir","rd","rmdir","type","tree","attrib","pushd","popd","start","shift","not","exist","defined","errorlevel","equ","neq","lss","leq","gtr","geq","tokens","delims","usebackq","skip","nul"],
    sql: ["SELECT","FROM","WHERE","INSERT","INTO","VALUES","UPDATE","SET","DELETE","CREATE","TABLE","ALTER","DROP","JOIN","LEFT","RIGHT","INNER","OUTER","ON","GROUP","BY","ORDER","LIMIT","OFFSET","AND","OR","NOT","NULL","PRIMARY","KEY","FOREIGN","REFERENCES","INDEX","DEFAULT","RETURNING","AS","DISTINCT","COUNT"],
    cpp: ["int","char","bool","void","float","double","long","short","unsigned","signed","struct","class","enum","union","namespace","using","template","typename","typedef","const","constexpr","static","inline","virtual","override","public","private","protected","friend","operator","new","delete","this","nullptr","true","false","auto","sizeof","return","if","else","for","while","do","switch","case","default","break","continue","goto","try","catch","throw","uint8_t","uint16_t","uint32_t","uint64_t","int8_t","int16_t","int32_t","int64_t","size_t"],
    graphql: ["type","query","mutation","subscription","input","enum","interface","union","scalar","schema","fragment","on","implements","extend","directive","true","false","null","ID","Int","Float","String","Boolean"],
    cypher: ["MATCH","OPTIONAL","WHERE","RETURN","CREATE","MERGE","DELETE","DETACH","SET","REMOVE","WITH","UNWIND","ORDER","BY","SKIP","LIMIT","AS","ON","DISTINCT","UNION","ALL","CALL","YIELD","FOREACH","CASE","WHEN","THEN","ELSE","END","AND","OR","XOR","NOT","IN","STARTS","ENDS","CONTAINS","IS","NULL","TRUE","FALSE","EXISTS","INDEX","CONSTRAINT","UNIQUE","REQUIRE","FOR","LOAD","CSV","HEADERS","USING","PERIODIC","COMMIT","ASC","DESC","DROP","SHOW","PROFILE","EXPLAIN","COUNT","COLLECT","node","relationship"],
  };

  const LANG_SET = {
    js:  new Set([...KEYWORDS.common, ...KEYWORDS.js]),
    ts:  new Set([...KEYWORDS.common, ...KEYWORDS.ts]),
    typescript: new Set([...KEYWORDS.common, ...KEYWORDS.ts]),
    tsx: new Set([...KEYWORDS.common, ...KEYWORDS.ts]),
    cpp: new Set(KEYWORDS.cpp),
    c:   new Set(KEYWORDS.cpp),
    "c++": new Set(KEYWORDS.cpp),
    py:  new Set([...KEYWORDS.py, "self","cls","print","str","int","float","dict","list","bool","tuple","set","range"]),
    python: new Set([...KEYWORDS.py, "self","cls"]),
    go:  new Set(KEYWORDS.go),
    rust: new Set(KEYWORDS.rust),
    rs:  new Set(KEYWORDS.rust),
    php: new Set([...KEYWORDS.php, ...KEYWORDS.common.filter(k=>!["this","super"].includes(k))]),
    bash: new Set(KEYWORDS.bash),
    sh:  new Set(KEYWORDS.bash),
    shell: new Set(KEYWORDS.bash),
    zsh: new Set(KEYWORDS.bash),
    batch: new Set(KEYWORDS.batch.map(k=>k.toLowerCase()).concat(KEYWORDS.batch.map(k=>k.toUpperCase()))),
    bat: new Set(KEYWORDS.batch.map(k=>k.toLowerCase()).concat(KEYWORDS.batch.map(k=>k.toUpperCase()))),
    cmd: new Set(KEYWORDS.batch.map(k=>k.toLowerCase()).concat(KEYWORDS.batch.map(k=>k.toUpperCase()))),
    sql: new Set(KEYWORDS.sql.map(k=>k.toLowerCase()).concat(KEYWORDS.sql)),
    graphql: new Set(KEYWORDS.graphql),
    gql: new Set(KEYWORDS.graphql),
    cypher: new Set(KEYWORDS.cypher.map(k=>k.toLowerCase()).concat(KEYWORDS.cypher)),
    cql: new Set(KEYWORDS.cypher.map(k=>k.toLowerCase()).concat(KEYWORDS.cypher)),
  };

  const HASH_COMMENT = new Set(["py","python","bash","sh","env","yaml","toml","ruby","elixir","dockerfile","http","properties","cmake","perl","r","makefile","gitignore"]);
  const esc = (s) => s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");

  function highlight(code, lang) {
    lang = (lang || "").toLowerCase();
    if (lang === "text" || lang === "plain" || lang === "") return esc(code);
    const kw = LANG_SET[lang] || new Set([...KEYWORDS.common]);
    const hashComment = HASH_COMMENT.has(lang);
    const batchLang = (lang === "batch" || lang === "bat" || lang === "cmd");

    // token master regex — order = priority
    const parts = [
      "(?<block>/\\*[\\s\\S]*?\\*/)",                          // block comment
      batchLang ? "(?<rem>::[^\\n]*|(?:^|\\n)[ \\t]*[Rr][Ee][Mm](?=[ \\t\\n]).*)" : null, // batch REM / :: comment
      hashComment ? "(?<hash>#.*)" : null,                     // # comment
      "(?<line>//.*)",                                         // // comment
      "(?<tstr>\"\"\"[\\s\\S]*?\"\"\"|'''[\\s\\S]*?''')",      // triple string (py)
      "(?<str>\"(?:\\\\.|[^\"\\\\])*\"|'(?:\\\\.|[^'\\\\])*'|`(?:\\\\.|[^`\\\\])*`)", // string
      "(?<attr>#\\[[^\\]]*\\]|@[A-Za-z_][\\w.]*)",            // rust attr / decorator
      "(?<num>\\b\\d[\\d_]*\\.?\\d*(?:[eE][+-]?\\d+)?\\b)",   // number
      batchLang ? "(?<bvar>%[^%\\n ]+%|%%~?[a-zA-Z]|%~[a-zA-Z0-9]+|%[*0-9]|![^!\\n ]+!)" : null, // batch %VAR% / !VAR! / %%i / %~dp0
      "(?<var>\\$[A-Za-z_]\\w*)",                              // php/bash var
      "(?<fn>[A-Za-z_]\\w*(?=\\s*\\())",                      // function call
      "(?<id>[A-Za-z_]\\w*)",                                  // identifier
      "(?<ws>\\s+)",
      "(?<other>[^\\s\\w])",
    ].filter(Boolean);

    const re = new RegExp(parts.join("|"), "g");
    let out = "", m;
    while ((m = re.exec(code)) !== null) {
      const g = m.groups;
      if (g.block || g.line || g.hash || g.rem) { out += `<span class="tok-com">${esc(m[0])}</span>`; }
      else if (g.tstr) { out += `<span class="tok-str">${esc(m[0])}</span>`; }
      else if (g.str) { out += `<span class="tok-str">${esc(m[0])}</span>`; }
      else if (g.attr) { out += `<span class="tok-fn">${esc(m[0])}</span>`; }
      else if (g.num) { out += `<span class="tok-num">${esc(m[0])}</span>`; }
      else if (g.bvar) { out += `<span class="tok-var">${esc(m[0])}</span>`; }
      else if (g.var) { out += `<span class="tok-var">${esc(m[0])}</span>`; }
      else if (g.fn) {
        const w = m[0];
        if (kw.has(w)) out += `<span class="tok-kw">${esc(w)}</span>`;
        else out += `<span class="tok-fn">${esc(w)}</span>`;
      }
      else if (g.id) {
        const w = m[0];
        if (kw.has(w)) out += `<span class="tok-kw">${esc(w)}</span>`;
        else if (/^[A-Z]/.test(w)) out += `<span class="tok-type">${esc(w)}</span>`;
        else out += esc(w);
      }
      else { out += esc(m[0]); }
    }
    return out;
  }

  window.Highlight = highlight;
})();
