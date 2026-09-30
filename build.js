#!/usr/bin/env node
// 学习笔记的构建脚本。
// 审核 words/、questions/、topics/ 中的中文稿与 en/ 下对应的英文稿，渲染成 HTML，写入 data.js。
//
//   node build.js          构建一次
//   node build.js -w       监视各目录，改动后重新构建
//   node build.js --fix    先修正中文稿的排版问题，再构建

'use strict';
const fs = require('fs');
const os = require('os');
const path = require('path');

const WORDS = path.join(__dirname, 'words');
const OUT = path.join(__dirname, 'data.js');
const TOC = path.join(__dirname, '目录.md');
const QUESTIONS = path.join(__dirname, 'questions');
const TOPICS = path.join(__dirname, 'topics');
const EN = path.join(__dirname, 'en');   // 英文稿：en/words、en/questions、en/topics，文件名与中文稿相同
const [EN_WORDS, EN_QUESTIONS, EN_TOPICS] = ['words', 'questions', 'topics'].map(d => path.join(EN, d));
const KATEX = path.join(__dirname, 'vendor', 'katex', 'katex.min.js');
const FETCH_KATEX = `mkdir -p vendor/katex && curl -sL "$(npm view katex dist.tarball)" | tar -xz -C vendor/katex --strip-components=2 package/dist/katex.min.js package/dist/katex.min.css 'package/dist/fonts/*.woff2'`;


// 准入规则中可由机器判断的部分。其余见 README。

const LIMIT = { chars: 250, paragraphs: 3, sentence: 60 };   // 均不计公式，也不计「注」
const REMARK = 200;                                          // 「注」的字数上限，限一段
const ANSWER = { ...LIMIT, chars: 500, paragraphs: 4 };      // 问题的回答可以长一些
const SECTION = 1200;                                        // 专题每节的字数上限

const DISCOURAGED = [
  ['口语冗词', ['其实', '就是', '简单来说', '换句话说', '总之']],
  ['模糊用语', ['显然', '众所周知', '基本上', '大概', '大致', '差不多', '某种意义上']],
  ['宜用无人称陈述', ['我们', '你', '您']],
];

// 英文稿按词计，上限约为中文字数的七成。
const LIMIT_EN = { words: 175, paragraphs: 3, sentence: 40 };
const REMARK_EN = 140;
const ANSWER_EN = { ...LIMIT_EN, words: 350, paragraphs: 4 };
const SECTION_EN = 840;

const DISCOURAGED_EN = [
  ['口语冗词', ['basically', 'actually', 'simply put', 'in other words']],
  ['模糊用语', ['obviously', 'of course', 'roughly speaking', 'kind of', 'sort of']],
  ['宜用无人称陈述', ['we', 'us', 'our', 'you', 'your']],
];
const CJK = /[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]/;   // 汉字与全角标点

const HAN = '[\\u3400-\\u4dbf\\u4e00-\\u9fff\\uf900-\\ufaff]';   // 汉字
const WEST = '[A-Za-z0-9\\ue100-\\ue7ff]';                         // 西文、数字、行内公式
const WIDE = /[\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef]/;     // 汉字与全角标点
const FULL = { ',': '，', '.': '。', ';': '；', ':': '：', '!': '！', '?': '？' };

// 排版规则：re 找出问题，when 进一步限定，fix 给出改法。按顺序应用。
const TYPESET = [
  {
    why: '中文语境宜用全角括号',
    re: /\(([^()\n]*)\)/g,
    when: (m, i, s) => new RegExp(HAN).test(`${s[i - 1]}${m}${s[i + m.length]}`),
    fix: m => `（${m.slice(1, -1)}）`,
  },
  {
    why: '汉字后宜用全角标点',
    re: new RegExp(`(?<=${HAN})[,.;:!?]|(?<=[\\ue100-\\ue7ff])[,.;:!?](?=${HAN}|[ \\t]*$)`, 'gm'),
    fix: m => FULL[m],
  },
  {
    why: '全角标点两侧不空格',
    re: /[ \t]+(?=[，。；：、？！）」』》])|(?<=[（「『《，。；：、？！])[ \t]+/g,
    fix: () => '',
  },
  {
    why: '汉字与西文、数字、公式之间宜空一格',
    re: new RegExp(`${HAN}(?=${WEST})|${WEST}(?=${HAN})`, 'g'),
    fix: m => `${m} `,
  },
  {
    why: '标点叠用',
    re: /([，。；：、？！])\1+/g,
    fix: m => m[0],
  },
];


// 语法。代码、公式、链接先换成私用区中的单个字符，其余文字才按规则处理。

const SYNTAX = {
  fence: /^```[^\n]*\n([\s\S]*?)\n```[ \t]*$/gm,
  code: /`([^`\n]+)`/g,
  dollar: /\\\$/g,
  display: /\$\$([\s\S]+?)\$\$/g,
  inline: /\$([^$\n]+?)\$/g,
  link: /\[\[([^\]|\n]+)(?:\|([^\]\n]+))?\]\]/g,
};

const INLINE = 0xe100, DISPLAY = 0xe800, LINK = 0xec00;   // 三类占位符在私用区中的起点
const held = () => /[\ue100-\uefff]/g;
const slot = c => {
  const n = c.charCodeAt(0);
  return n - (n >= LINK ? LINK : n >= DISPLAY ? DISPLAY : INLINE);
};

const esc = s => s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const emphasis = s => s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/\*(.+?)\*/g, '<em>$1</em>');
const size = s => [...s.replace(/\s/g, '')].length;
const wordCount = s => s.split(/\s+/).filter(w => /[\p{L}\p{N}]/u.test(w)).length;

// 长路径与长标识符的折行点：分隔符之后，以及驼峰的词与词之间。
const breakCode = s => s.replace(/([a-z0-9])(?=[A-Z])|([A-Z])(?=[A-Z][a-z])/g, '$1$2<wbr>').replace(/([/.:_])/g, '$1<wbr>');

// edges 为真时，链接两侧各留一份首尾可见字的副本，供排版检查判断链接挨着什么字。
function mask(src, edges = false) {
  const spans = [];
  const hold = (base, span) => String.fromCharCode(base + spans.push(span) - 1);
  const raw = s => s.replace(held(), c => spans[slot(c)].raw);
  const text = src
    .replace(SYNTAX.fence, (r, code) => hold(DISPLAY, { raw: r, text: '', block: true, html: `<pre><code>${esc(code)}</code></pre>` }))
    .replace(SYNTAX.code, (r, code) => hold(INLINE, { raw: r, text: code, code, html: `<code>${breakCode(esc(code))}</code>` }))
    .replace(SYNTAX.dollar, r => hold(INLINE, { raw: r, text: '$', html: '$' }))
    .replace(SYNTAX.display, (r, tex) => hold(DISPLAY, { raw: raw(r), tex: raw(tex), display: true }))
    .replace(SYNTAX.inline, (r, tex) => hold(INLINE, { raw: raw(r), tex: raw(tex), display: false }))
    .replace(SYNTAX.link, (r, target, label = target) => {
      const c = hold(LINK, { raw: r, target: raw(target).trim(), label: label.trim() });
      const seen = [...label.trim().replace(/\*/g, '')];
      return edges ? seen[0] + c + seen[seen.length - 1] : c;
    });
  return { text, spans };
}

function unmask(text, spans, edges = false) {
  if (edges) text = text.replace(/[^]([\uec00-\uefff])[^]/g, '$1');
  return text
    .replace(/[\uec00-\uefff]/g, c => spans[slot(c)].raw)
    .replace(held(), c => spans[slot(c)].raw);
}

// 纯文本：链接取显示的文字，代码取原文，公式不计。
function plain(text, spans) {
  return text
    .replace(/[\uec00-\uefff]/g, c => spans[slot(c)].label)
    .replace(held(), c => spans[slot(c)].text ?? '')
    .replace(/\*/g, '');
}

// 按空行分段。段内换行两侧都是汉字或全角标点时删去，否则作一个空格。
// 正文、注、例：以「注：」起头的段落是注，以「例：」起头的段落及其后各段是例；英文稿用「Note:」「Example:」。
const MARKS = {
  zh: { note: '注：', ex: '例：', labels: ['注', '例'] },
  en: { note: 'Note:', ex: 'Example:', labels: ['Note', 'Example'] },
};
function parts3(ps, mark = MARKS.zh) {
  const ex = ps.findIndex(p => p.startsWith(mark.ex));
  const body = ex === -1 ? ps : ps.slice(0, ex);
  const note = body.findIndex(p => p.startsWith(mark.note));
  return {
    def: note === -1 ? body : body.slice(0, note),
    note: note === -1 ? [] : body.slice(note),
    ex: ex === -1 ? [] : ps.slice(ex),
  };
}

const joinLines = p => p.replace(/[ \t]*\n[ \t]*/g, (m, i, s) => WIDE.test(s[i - 1]) && WIDE.test(s[i + m.length]) ? '' : ' ');
function paragraphs(text, join = true) {
  return text.split(/\n[ \t]*\n/)
    .map(p => join ? joinLines(p.trim()) : p.trim())
    .filter(Boolean);
}

// 专题中的一块：代码块、列表、表格或段落。
function topicBlock(p, inline, spans) {
  const lines = p.split('\n').map(l => l.trim());
  const only = p.length === 1 && spans[slot(p)]?.block;
  if (only) return spans[slot(p)].html;
  if (lines.every(l => l.startsWith('- '))) return `<ul>${lines.map(l => `<li>${inline(l.slice(2))}</li>`).join('')}</ul>`;
  if (lines.every(l => /^\d+\.\s/.test(l))) return `<ol>${lines.map(l => `<li>${inline(l.replace(/^\d+\.\s+/, ''))}</li>`).join('')}</ol>`;
  if (lines.length >= 2 && lines.every(l => l.startsWith('|')) && /^\|[\s:|-]+\|$/.test(lines[1])) {
    const cells = l => l.replace(/^\||\|$/g, '').split('|').map(c => c.trim());
    const row = (l, tag) => `<tr>${cells(l).map(c => `<${tag}>${inline(c)}</${tag}>`).join('')}</tr>`;
    return `<div class="table"><table><thead>${row(lines[0], 'th')}</thead><tbody>${lines.slice(2).map(l => row(l, 'td')).join('')}</tbody></table></div>`;
  }
  return `<p>${inline(joinLines(p))}</p>`;
}

// 首句，用于目录。遇到句号或独立公式即止。
function firstSentence(text) {
  const ps = paragraphs(text);
  const p = ps[0] ?? '';
  const k = p.search(/[。\ue800-\uebff]/);
  if (k === -1) return ps.length > 1 ? `${p}…` : p;
  return p[k] === '。' ? p.slice(0, k + 1) : `${p.slice(0, k).trim()}…`;
}

function renderer(spans, has, anchors = true) {
  const inline = s => emphasis(esc(s)).replace(held(), c => {
    const span = spans[slot(c)];
    if (span.target === undefined) return span.html;
    const label = inline(span.label);
    if (!anchors) return label;
    const cls = has(span.target) ? 'w' : 'w missing';
    return `<a class="${cls}" href="#${encodeURIComponent(span.target)}" data-w="${esc(span.target)}">${label}</a>`;
  });
  return inline;
}


// 审核

// 行内公式中顶层的逗号后允许折行，免得长元组整个挤到下一行、把上一行的字距拉开。
const breakable = tex => {
  let depth = 0, out = '';
  for (const ch of tex) {
    if (ch === '{') depth++;
    else if (ch === '}') depth--;
    out += ch === ',' && depth === 0 ? ',\\allowbreak ' : ch;
  }
  return out;
};

function compile(span, errors, hints) {
  try {
    span.html = katex.renderToString(span.display ? span.tex : breakable(span.tex), {
      displayMode: span.display,
      throwOnError: true,
      strict: (code, msg) => {
        hints.push(`「${span.raw}」：${code === 'unicodeTextInMathMode' ? '公式中的文字宜放进 \\text{}' : `不合 LaTeX 规范（${msg}）`}`);
        return 'ignore';
      },
    }).replaceAll('\\allowbreak ', '');
  } catch (e) {
    const why = String(e.message).replace(/^KaTeX parse error: /, '').replace(/ at (position|end of input)[^]*$/, '');
    errors.push(`「${span.raw}」：公式无法解析（${why}）`);
  }
}

// 取出问题附近的原文，不把链接切断。
function around(text, i, j) {
  const isLink = k => /[\uec00-\uefff]/.test(text[k] ?? '');
  i = Math.max(0, i - 3);
  j = Math.min(text.length, j + 3);
  if (isLink(i)) i -= 1; else if (isLink(i - 1)) i -= 2;
  if (isLink(j - 1)) j += 1; else if (isLink(j)) j += 2;
  return text.slice(Math.max(0, i), j);
}

function typesetHints(src) {
  const { text, spans } = mask(src, true);
  const hints = [];
  for (const rule of TYPESET) {
    const found = [...text.matchAll(rule.re)].filter(m => !rule.when || rule.when(m[0], m.index, text));
    if (!found.length) continue;
    const [m] = found;
    const snippet = unmask(around(text, m.index, m.index + m[0].length), spans, true).replace(/\s+/g, ' ').trim();
    hints.push(`「${snippet}」${found.length > 1 ? `等 ${found.length} 处` : ''}：${rule.why}`);
  }
  return hints;
}

function typeset(src) {
  const { text, spans } = mask(src, true);
  let t = text;
  for (const rule of TYPESET) {
    t = t.replace(rule.re, (m, ...rest) => {
      const [i, s] = rest.slice(-2);
      return !rule.when || rule.when(m, i, s) ? rule.fix(m) : m;
    });
  }
  return unmask(t, spans, true);
}

// 审核一条词条，或一个问题的回答（question 为真）。
function audit(name, src, question = false) {
  const errors = [], hints = [];
  const limit = question ? ANSWER : LIMIT;
  const { text, spans } = mask(src);
  const body = plain(text, spans);

  if (!src.trim()) errors.push('内容为空');
  if (/^#{1,6}\s/m.test(text)) errors.push(question ? '不能使用标题' : '不能使用标题：一个词条只有一段解释');
  if (/!\[/.test(text)) errors.push('不能插入图片');
  if (spans.some(s => s.block)) errors.push('不能使用代码块：代码只宜写进专题');
  const real = p => /[^\s\ue800-\uebff.,。，]/.test(p);   // 不只是一个独立公式的段落
  const { def, note, ex } = parts3(paragraphs(text));
  const main = question ? [...def, ...note, ...ex] : [...def, ...ex];
  const mainSize = size(plain(main.join('\n'), spans));
  if (mainSize > limit.chars) errors.push(`正文 ${mainSize} 字，超过上限 ${limit.chars} 字`);
  if (main.filter(real).length > limit.paragraphs) errors.push(`共 ${main.filter(real).length} 段，超过上限 ${limit.paragraphs} 段`);
  if (!question && note.length) {
    const noteSize = size(plain(note.join('\n'), spans)) - 2;   // 不计「注：」二字
    if (noteSize > REMARK) errors.push(`「注」${noteSize} 字，超过上限 ${REMARK} 字`);
    if (note.filter(real).length > 1) errors.push('「注」只能有一段');
  }
  const links = spans.filter(s => s.target !== undefined).map(s => s.target);
  if (links.includes(name)) errors.push('链接到了自身');
  if (question) {
    if (!name.endsWith('？')) errors.push('标题须是问句，以「？」结尾');
    if (!links.length) errors.push('没有引用任何词条：回答须由词条组成');
    if (/\*\*.+?\*\*/.test(text)) errors.push('不能用黑体引入新术语：新术语应先写成词条');
  }
  for (const s of spans) if (s.tex !== undefined) compile(s, errors, hints);

  for (const [why, list] of DISCOURAGED) for (const w of list) if (body.includes(w)) hints.push(`「${w}」：${why}`);
  for (const q of new Set(body.match(/[？！?!]/g))) hints.push(`「${q}」：释义宜用陈述句`);
  const examples = (body.match(/例：/g) ?? []).length;
  if (examples > 1) hints.push(`「例：」出现 ${examples} 次：例子至多一个`);
  if (examples && !paragraphs(text).some(p => p.startsWith('例：'))) hints.push('「例：」：例子宜另起一段');
  const remarks = (body.match(/注：/g) ?? []).length;
  if (remarks > 1) hints.push(`「注：」出现 ${remarks} 次：注至多一段`);
  if (remarks && !note.length) hints.push('「注：」：注宜另起一段，并放在例之前');
  const bare = text.replace(held(), c => spans[slot(c)].target !== undefined ? spans[slot(c)].label : '').replace(/\*/g, '');   // 句长不计公式与代码
  for (const s of bare.split(/[。！？]|\n\s*\n/)) {
    if (size(s) > limit.sentence) hints.push(`「${s.replace(/\s+/g, ' ').trim().slice(0, 12)}…」：句子 ${size(s)} 字，宜拆分`);
  }
  const typo = typesetHints(src);

  return { text, spans, links, errors, hints: [...hints, ...typo], typo: typo.length > 0 };
}

// 专题开头的「版本：」「源码：」两行。
function topicHead(src) {
  const lines = src.split('\n');
  const head = {};
  let k = 0;
  for (; k < lines.length && /^(版本|源码)：/.test(lines[k].trim()); k++) {
    const [key, ...rest] = lines[k].trim().split('：');
    head[key] = rest.join('：').trim();
  }
  return { head, body: lines.slice(k).join('\n').trim() };
}

// 核对反引号中的源码路径；写成「路径:符号」时，另核对该符号出现在文件中。
function checkSource(spans, root, errors, hints) {
  if (!root) return;
  if (!fs.existsSync(root)) { hints.push(`源码目录 ${root} 不存在，未核对源码引用`); return; }
  for (const s of spans) {
    const m = s.code?.match(/^([\w.-]+(?:\/[\w.-]+)+)(?::([\w.]+))?$/);
    if (!m) continue;
    const file = path.join(root, m[1]);
    if (!fs.existsSync(file)) errors.push(`「${m[1]}」：源码中没有这个路径`);
    else if (m[2] && !(fs.statSync(file).isFile() && fs.readFileSync(file, 'utf8').includes(m[2].split('.').pop()))) {
      errors.push(`「${s.code}」：该文件中找不到 ${m[2]}`);
    }
  }
}

function auditTopic(name, src) {
  const errors = [], hints = [];
  const { head, body: rest } = topicHead(src);
  if (!head['版本']) errors.push('须以「版本：」一行开头，注明所依据的版本');
  const { text, spans } = mask(rest);
  if (/!\[/.test(text)) errors.push('不能插入图片');
  if (/^(#|#{3,6})\s/m.test(text)) errors.push('只能用「## 」一级小节');
  const parts = text.split(/^## +(.+)$/m);   // 导言、标题 1、正文 1、标题 2……
  if (parts.length < 3) errors.push('至少要有一个以「## 」起头的小节');
  // 小节字数只计文字：公式与代码不计，链接计显示的文字。
  const bare = t => t.replace(held(), c => spans[slot(c)].target !== undefined ? spans[slot(c)].label : '').replace(/\*/g, '');
  for (let i = 1; i < parts.length; i += 2) {
    const n = size(bare(parts[i + 1]));
    if (n > SECTION) errors.push(`「${plain(parts[i], spans).trim()}」一节 ${n} 字，超过上限 ${SECTION} 字`);
  }
  for (const s of spans) if (s.tex !== undefined) compile(s, errors, hints);
  const links = spans.filter(s => s.target !== undefined).map(s => s.target);
  checkSource(spans, head['源码']?.replace(/^~(?=\/)/, os.homedir()), errors, hints);

  const bodies = parts.filter((_, i) => i % 2 === 0).join('\n\n');
  const prose = plain(bodies, spans);
  for (const [why, list] of DISCOURAGED) for (const w of list) if (prose.includes(w)) hints.push(`「${w}」：${why}`);
  for (const q of new Set(prose.match(/[？！?!]/g))) hints.push(`「${q}」：正文宜用陈述句`);
  for (const line of paragraphs(bodies, false).flatMap(p => /^(- |\d+\.\s|\|)/m.test(p) ? p.split('\n') : [joinLines(p)])) {
    for (const s of bare(line).split(/[。！？]/)) {
      if (size(s) > LIMIT.sentence) hints.push(`「${s.replace(/\s+/g, ' ').trim().slice(0, 12)}…」：句子 ${size(s)} 字，宜拆分`);
    }
  }
  const typo = typesetHints(rest);
  return { text, spans, links, errors, hints: [...hints, ...typo], typo: typo.length > 0, head, parts };
}

// 英文稿。准入规则与中文稿相同，只是按词计数；排版另查是否夹有汉字或全角标点。

// 链接可写英文名，也可写中文词条名，一律解析成中文词条名；未写显示文字而写了中文名时，显示英文名。
function resolveLinks(spans, lookup, enName) {
  for (const s of spans) {
    if (s.target === undefined) continue;
    const id = lookup.get(s.target.toLowerCase());
    if (!id) continue;
    if (s.label === s.target && CJK.test(s.target)) s.label = enName(id);
    s.target = id;
  }
}

// 英文稿应与中文稿引用同一组词条，路线与预备才两边一致。
function sameLinks(en, zh, hints) {
  const a = new Set(en), b = new Set(zh);
  const fewer = [...b].filter(t => !a.has(t)), more = [...a].filter(t => !b.has(t));
  if (fewer.length) hints.push(`比中文稿少了链接：${fewer.map(t => `「${t}」`).join('')}`);
  if (more.length) hints.push(`比中文稿多了链接：${more.map(t => `「${t}」`).join('')}`);
}

function proseHints(body, hints) {
  for (const [why, list] of DISCOURAGED_EN) for (const w of list) if (new RegExp(`\\b${w}\\b`, 'i').test(body)) hints.push(`「${w}」：${why}`);
  const m = body.match(new RegExp(`.{0,8}${CJK.source}.{0,8}`));
  if (m) hints.push(`「${m[0].trim()}」：英文稿中夹有汉字或全角标点`);
}

const sentencesEn = t => t.split(/(?<=[.!?])\s+(?=[A-Z0-9\ue100-\uefff“(])|\n\s*\n/);
function sentenceHints(t, max, hints) {
  for (const s of sentencesEn(t)) {
    const n = wordCount(s);
    if (n > max) hints.push(`「${s.replace(/\s+/g, ' ').trim().slice(0, 24)}…」：句子 ${n} 词，宜拆分`);
  }
}

// 问题与专题的英文稿以「# 」一行给出英文标题。
function titled(src, errors) {
  const m = src.match(/^#\s+(.+)\n+([^]*)$/);
  if (!m) errors.push('须以「# 」一行给出英文标题');
  return m ? { title: m[1].trim(), body: m[2] } : { title: '', body: src };
}

function auditEn(name, src, { question = false, lookup, enName, zhLinks }) {
  const errors = [], hints = [];
  const limit = question ? ANSWER_EN : LIMIT_EN;
  const { title, body: raw } = question ? titled(src, errors) : { title: '', body: src };
  const { text, spans } = mask(raw);
  resolveLinks(spans, lookup, enName);
  const body = plain(text, spans);

  if (!raw.trim()) errors.push('内容为空');
  if (/^#{1,6}\s/m.test(text)) errors.push(question ? '不能使用标题' : '不能使用标题：一个词条只有一段解释');
  if (/!\[/.test(text)) errors.push('不能插入图片');
  if (spans.some(s => s.block)) errors.push('不能使用代码块：代码只宜写进专题');
  const real = p => /[^\s\ue800-\uebff.,]/.test(p);
  const { def, note, ex } = parts3(paragraphs(text), MARKS.en);
  const main = question ? [...def, ...note, ...ex] : [...def, ...ex];
  const n = wordCount(plain(main.join('\n'), spans));
  if (n > limit.words) errors.push(`正文 ${n} 词，超过上限 ${limit.words} 词`);
  if (main.filter(real).length > limit.paragraphs) errors.push(`共 ${main.filter(real).length} 段，超过上限 ${limit.paragraphs} 段`);
  if (!question && note.length) {
    const k = wordCount(plain(note.join('\n'), spans)) - 1;   // 不计「Note:」
    if (k > REMARK_EN) errors.push(`「Note」${k} 词，超过上限 ${REMARK_EN} 词`);
    if (note.filter(real).length > 1) errors.push('「Note」只能有一段');
  }
  const links = spans.filter(s => s.target !== undefined).map(s => s.target);
  if (links.includes(name)) errors.push('链接到了自身');
  if (question) {
    if (title && !title.endsWith('?')) errors.push('英文标题须是问句，以「?」结尾');
    if (!links.length) errors.push('没有引用任何词条：回答须由词条组成');
    if (/\*\*.+?\*\*/.test(text)) errors.push('不能用黑体引入新术语：新术语应先写成词条');
  }
  for (const s of spans) if (s.tex !== undefined) compile(s, errors, hints);
  for (const t of new Set(links)) if (!lookup.has(t.toLowerCase())) hints.push(`「${t}」：尚无此词条`);

  proseHints(body, hints);
  if (!question && body.includes('?')) hints.push('「?」：释义宜用陈述句');
  if ((body.match(/\bExample:/g) ?? []).length > 1) hints.push('「Example:」出现多次：例子至多一个');
  if ((body.match(/\bNote:/g) ?? []).length > 1) hints.push('「Note:」出现多次：注至多一段');
  sentenceHints(text.replace(held(), c => spans[slot(c)].target !== undefined ? spans[slot(c)].label : '').replace(/\*/g, ''), limit.sentence, hints);
  sameLinks(links, zhLinks, hints);
  return { title, text, spans, links, errors, hints };
}

function auditTopicEn(name, src, { lookup, enName, zhLinks, root }) {
  const errors = [], hints = [];
  const { title, body: raw } = titled(src, errors);
  const { text, spans } = mask(raw);
  resolveLinks(spans, lookup, enName);
  if (/!\[/.test(text)) errors.push('不能插入图片');
  if (/^(#|#{3,6})\s/m.test(text)) errors.push('只能用「## 」一级小节');
  const parts = text.split(/^## +(.+)$/m);
  if (parts.length < 3) errors.push('至少要有一个以「## 」起头的小节');
  const bare = t => t.replace(held(), c => spans[slot(c)].target !== undefined ? spans[slot(c)].label : '').replace(/\*/g, '');
  for (let i = 1; i < parts.length; i += 2) {
    const n = wordCount(bare(parts[i + 1]));
    if (n > SECTION_EN) errors.push(`「${plain(parts[i], spans).trim()}」一节 ${n} 词，超过上限 ${SECTION_EN} 词`);
  }
  for (const s of spans) if (s.tex !== undefined) compile(s, errors, hints);
  const links = spans.filter(s => s.target !== undefined).map(s => s.target);
  for (const t of new Set(links)) if (!lookup.has(t.toLowerCase())) hints.push(`「${t}」：尚无此词条`);
  checkSource(spans, root, errors, hints);

  const bodies = parts.filter((_, i) => i % 2 === 0).join('\n\n');
  proseHints(plain(bodies, spans), hints);
  for (const line of paragraphs(bodies, false).flatMap(p => /^(- |\d+\.\s|\|)/m.test(p) ? p.split('\n') : [joinLines(p)])) {
    sentenceHints(bare(line), LIMIT_EN.sentence, hints);
  }
  sameLinks(links, zhLinks, hints);
  return { title, text, spans, links, errors, hints, parts };
}

// 互相引用的词条组，即引用关系图中的强连通分量（Tarjan 算法）。
function cycles(entries) {
  const index = new Map(), low = new Map(), stack = [], groups = [];
  const visit = v => {
    index.set(v, index.size);
    low.set(v, index.get(v));
    stack.push(v);
    for (const w of new Set(entries.get(v).links)) {
      if (!entries.has(w)) continue;
      if (!index.has(w)) {
        visit(w);
        low.set(v, Math.min(low.get(v), low.get(w)));
      } else if (stack.includes(w)) {
        low.set(v, Math.min(low.get(v), index.get(w)));
      }
    }
    if (low.get(v) === index.get(v)) {
      const group = stack.splice(stack.indexOf(v));
      if (group.length > 1) groups.push(group);
    }
  };
  for (const v of entries.keys()) if (!index.has(v)) visit(v);
  return groups;
}


// 章节。目录.md 中以「# 」起头的行是章名，其下每行一个词条；章与词条的英文名都写在其后的全角括号中。
function readToc() {
  if (!fs.existsSync(TOC)) return [];
  const chapters = [];
  for (const raw of fs.readFileSync(TOC, 'utf8').split(/\r?\n/)) {
    const line = raw.trim().normalize('NFC');
    if (!line) continue;
    const [, name, en] = line.replace(/^# /, '').match(/^(.+?)(?:（(.+)）)?$/);
    if (line.startsWith('# ')) { chapters.push({ title: name.trim(), en: en?.trim() ?? '', items: [] }); continue; }
    if (!chapters.length) chapters.push({ title: '', en: '', items: [] });
    chapters[chapters.length - 1].items.push({ name: name.trim(), en: en?.trim() ?? '' });
  }
  return chapters;
}

// 章内次序：依给定次序，但每条排在它链接到的同章词条之后（深度优先的后序）。
function chapterOrder(admitted, names) {
  const inside = new Set(names), out = [], seen = new Set();
  const visit = n => {
    if (seen.has(n)) return;
    seen.add(n);
    for (const t of admitted.get(n).links) if (inside.has(t)) visit(t);
    out.push(n);
  };
  names.forEach(visit);
  return out;
}

// 按目录分章。未列入目录的词条归入末尾一章，其中不被引用的词条先行。
function arrange(admitted, toc) {
  const home = new Map(), en = new Map(), hints = [];
  toc.forEach((c, i) => c.items.forEach(item => {
    if (home.has(item.name)) hints.push(`「${item.name}」：在目录中重复出现`);
    else if (!admitted.has(item.name)) hints.push(`「${item.name}」：尚无此词条，或未收录`);
    else home.set(item.name, i);
    if (item.en) en.set(item.name, item.en);
  }));
  const groups = toc.map((c, i) => ({ title: c.title, en: c.en, names: c.items.map(x => x.name).filter(n => home.get(n) === i) }));
  const rest = [...admitted.keys()].filter(n => !home.has(n));
  if (rest.length) {
    const cited = new Set(rest.flatMap(n => admitted.get(n).links));
    groups.push({ title: toc.length ? '未分章' : '', en: toc.length ? 'Unassigned' : '', names: [...rest.filter(n => !cited.has(n)), ...rest] });
  }
  const chapters = groups
    .filter(g => g.names.length)
    .map(g => ({ title: g.title, en: g.en, words: chapterOrder(admitted, [...new Set(g.names)]) }));
  return { chapters, en, hints, unlisted: toc.length ? rest : [] };
}


// 构建

const files = (dir = WORDS) => fs.existsSync(dir) ? fs.readdirSync(dir).filter(f => f.endsWith('.md') && !/^[_.]/.test(f)) : [];
const read = (f, dir = WORDS) => fs.readFileSync(path.join(dir, f), 'utf8').replace(/\r\n?/g, '\n').trim();

// 正文、注、例各成一块，首段以标签起头。
function blocks(e, has, label, lang = 'zh') {
  const mark = MARKS[lang];
  const inline = renderer(e.spans, has);
  const { def, note, ex } = parts3(paragraphs(e.text), mark);
  const block = (list, tag, strip) => list
    .map((p, j) => j ? `<p>${inline(p)}</p>` : `<p><span class="label">${tag}</span>${inline(strip ? p.slice(strip.length).trimStart() : p)}</p>`)
    .join('');
  return { def: block(def, label, ''), note: block(note, mark.labels[0], mark.note), ex: block(ex, mark.labels[1], mark.ex) };
}
const nameOf = f => path.basename(f, '.md').normalize('NFC').trim();

function build() {
  const collate = new Intl.Collator('zh').compare;
  const entries = files()
    .map(f => ({ name: nameOf(f), ...audit(nameOf(f), read(f)) }))
    .sort((a, b) => collate(a.name, b.name));

  const admitted = new Map(entries.filter(e => !e.errors.length).map(e => [e.name, e]));
  for (const e of entries) {
    for (const t of new Set(e.links)) if (t !== e.name && !admitted.has(t)) e.hints.push(`「${t}」：尚无此词条`);
  }
  for (const group of cycles(admitted)) {
    group.sort(collate);
    admitted.get(group[0]).hints.push(`${group.map(n => `「${n}」`).join('')}：互相引用，可能构成循环定义`);
  }

  const toc = readToc();
  const { chapters, en, hints: tocHints, unlisted } = arrange(admitted, toc);
  const order = chapters.flatMap(c => c.words);
  const pos = new Map(order.map((n, i) => [n, i]));
  for (const n of unlisted) admitted.get(n).hints.push('未列入目录.md');
  for (const [name, e] of admitted) {
    for (const t of new Set(e.links)) if (pos.get(t) > pos.get(name)) e.hints.push(`「${t}」：在目录中排在本条之后`);
  }

  // 预备：读懂一条所需的全部词条，即它直接或间接链接到的词条，按编号排列。
  const direct = n => [...new Set(admitted.get(n).links)].filter(t => t !== n && admitted.has(t));
  const prerequisites = n => {
    const seen = new Set();
    const walk = m => { for (const t of direct(m)) if (t !== n && !seen.has(t)) { seen.add(t); walk(t); } };
    walk(n);
    return [...seen].sort((a, b) => pos.get(a) - pos.get(b));
  };

  const has = w => admitted.has(w);
  const numbered = toc.length > 0;
  const words = {}, back = {};
  chapters.forEach((c, ci) => c.words.forEach((name, k) => {
    const e = admitted.get(name);
    words[name] = {
      n: numbered ? `${ci + 1}.${k + 1}` : `${pos.get(name) + 1}`,
      ch: ci,
      en: en.get(name) ?? '',
      ...blocks(e, has, '定义'),
      excerpt: renderer(e.spans, has, false)(firstSentence(e.text)),
      text: plain(e.text, e.spans).replace(/\s+/g, ' ').trim(),
      pre: prerequisites(name),
      direct: direct(name),
    };
    for (const t of new Set(e.links)) (back[t] ||= []).push(name);
  }));

  // 问题：回答中的链接按出现先后即为路线；问题按所需最后一条词条的编号排列，先易后难。
  const asked = files(QUESTIONS)
    .map(f => ({ name: nameOf(f), ...audit(nameOf(f), read(f, QUESTIONS), true) }))
    .sort((a, b) => collate(a.name, b.name));
  for (const q of asked) for (const t of new Set(q.links)) if (!admitted.has(t)) q.hints.push(`「${t}」：尚无此词条`);
  const questions = asked
    .filter(q => !q.errors.length)
    .map(q => {
      const route = [...new Set(q.links)].filter(has);
      const pre = [...new Set(route.flatMap(t => words[t].pre))]
        .filter(t => !route.includes(t))
        .sort((a, b) => pos.get(a) - pos.get(b));
      const depth = Math.max(...route.map(t => pos.get(t)));
      return { q: q.name, depth, ch: words[order[depth]].ch, route, pre, ...blocks(q, has, '答'), text: plain(q.text, q.spans).replace(/\s+/g, ' ').trim() };
    })
    .sort((a, b) => a.depth - b.depth || collate(a.q, b.q))
    .map(({ depth, ...q }) => q);   // ch：所需最深的一章，首页据此分组

  // 探索以问题为入口：每条词条都应至少被一个问题引用。
  if (questions.length) {
    const cited = new Set(questions.flatMap(q => q.route));
    for (const [name, e] of admitted) if (!cited.has(name)) e.hints.push('尚无问题引用本条');
  }

  // 专题：按标题排列；引用的词条按出现先后列出。
  const written = files(TOPICS)
    .map(f => ({ name: nameOf(f), ...auditTopic(nameOf(f), read(f, TOPICS)) }))
    .sort((a, b) => collate(a.name, b.name));
  for (const t of written) for (const w of new Set(t.links)) if (!admitted.has(w)) t.hints.push(`「${w}」：尚无此词条`);
  const topics = written.filter(t => !t.errors.length).map(t => {
    const inline = renderer(t.spans, has);
    const render = body => paragraphs(body, false).map(p => topicBlock(p, inline, t.spans)).join('');
    const sections = [];
    for (let i = 1; i < t.parts.length; i += 2) sections.push({ h: inline(t.parts[i].trim()), html: render(t.parts[i + 1]) });
    const route = [...new Set(t.links)].filter(has);
    return { t: t.name, version: t.head['版本'], sourced: Boolean(t.head['源码']), intro: render(t.parts[0]), sections, route, text: plain(t.text, t.spans).replace(/\s+/g, ' ').trim() };
  });

  // 英文版：与中文稿逐一配对。缺英文稿，或英文稿未通过审核时，网页上仍显示中文稿。
  const cjk = s => CJK.test(s);
  const enName = n => cjk(n) ? en.get(n) || n : n;                                     // 正文中的英文名
  const cap = s => /^[a-z]+(?![A-Za-z])/.test(s) ? s[0].toUpperCase() + s.slice(1) : s;   // 作标题时首字母大写，vLLM 之类不动
  const lookup = new Map();
  for (const n of admitted.keys()) {
    lookup.set(n.toLowerCase(), n);
    if (en.get(n)) lookup.set(en.get(n).toLowerCase(), n);
  }
  const enAudits = [], missing = [];
  const pair = (dir, name) => {
    const file = `${name}.md`;
    if (fs.existsSync(path.join(dir, file))) return read(file, dir);
    missing.push(name);
    return null;
  };
  const enWords = {};
  for (const name of order) {
    const src = pair(EN_WORDS, name);
    if (src === null) continue;
    const e = { name: `${name}（英文）`, ...auditEn(name, src, { lookup, enName, zhLinks: admitted.get(name).links }) };
    enAudits.push(e);
    if (!e.errors.length) enWords[name] = { title: cap(enName(name)), sub: cjk(name) ? name : en.get(name) ?? '', ...blocks(e, has, 'Definition', 'en'), text: plain(e.text, e.spans).replace(/\s+/g, ' ').trim() };
  }
  const enQuestions = {};
  for (const q of questions) {
    const src = pair(EN_QUESTIONS, q.q);
    if (src === null) continue;
    const zh = asked.find(x => x.name === q.q);
    const e = { name: `${q.q}（英文）`, ...auditEn(q.q, src, { question: true, lookup, enName, zhLinks: zh.links }) };
    enAudits.push(e);
    if (!e.errors.length) enQuestions[q.q] = { q: e.title, ...blocks(e, has, 'Answer', 'en'), text: plain(e.text, e.spans).replace(/\s+/g, ' ').trim() };
  }
  const enTopics = {};
  for (const t of topics) {
    const src = pair(EN_TOPICS, t.t);
    if (src === null) continue;
    const zh = written.find(x => x.name === t.t);
    const root = zh.head['源码']?.replace(/^~(?=\/)/, os.homedir());
    const e = { name: `${t.t}（英文）`, ...auditTopicEn(t.t, src, { lookup, enName, zhLinks: zh.links, root }) };
    enAudits.push(e);
    if (e.errors.length) continue;
    const inline = renderer(e.spans, has);
    const render = body => paragraphs(body, false).map(p => topicBlock(p, inline, e.spans)).join('');
    const sections = [];
    for (let i = 1; i < e.parts.length; i += 2) sections.push({ h: inline(e.parts[i].trim()), html: render(e.parts[i + 1]) });
    enTopics[t.t] = { t: e.title, intro: render(e.parts[0]), sections, text: plain(e.text, e.spans).replace(/\s+/g, ' ').trim() };
  }
  const known = new Set([...order, ...questions.map(q => q.q), ...topics.map(t => t.t)]);
  const orphans = [EN_WORDS, EN_QUESTIONS, EN_TOPICS].flatMap(d => files(d).map(nameOf)).filter(n => !known.has(n));
  const enNotes = { name: '英文版', errors: [], hints: [
    ...(missing.length ? [`尚无英文稿：${missing.map(n => `「${n}」`).join('')}`] : []),
    ...(orphans.length ? [`没有对应的中文稿：${orphans.map(n => `「${n}」`).join('')}`] : []),
  ] };

  const lines = order.map(n => `  ${JSON.stringify(n)}: ${JSON.stringify(words[n])}`);
  const qlines = questions.map(q => `  ${JSON.stringify(q)}`);
  const tlines = topics.map(t => `  ${JSON.stringify(t)}`);
  const english = `{\n  words: ${JSON.stringify(enWords)},\n  questions: ${JSON.stringify(enQuestions)},\n  topics: ${JSON.stringify(enTopics)}\n }`;
  fs.writeFileSync(OUT, `// 由 build.js 生成，请勿手改。\nwindow.WIKI = {\n chapters: ${JSON.stringify(chapters)},\n words: {\n${lines.join(',\n')}\n },\n questions: [\n${qlines.join(',\n')}\n ],\n topics: [\n${tlines.join(',\n')}\n ],\n back: ${JSON.stringify(back)},\n en: ${english}\n};\n`);

  const count = { entries: admitted.size, questions: questions.length, topics: topics.length };
  const enCount = { entries: Object.keys(enWords).length, questions: Object.keys(enQuestions).length, topics: Object.keys(enTopics).length };
  return report([...entries, ...asked, ...written, ...enAudits, enNotes], count, tocHints, enCount);
}

const tally = c => `${c.entries} 条${c.questions ? `、问题 ${c.questions} 个` : ''}${c.topics ? `、专题 ${c.topics} 篇` : ''}`;

function report(entries, count, tocHints = [], enCount = null) {
  let rejected = 0, hinted = 0, typo = false;
  if (tocHints.length) {
    console.log('目录.md');
    for (const m of [...new Set(tocHints)]) console.log(`  提示　${m}`);
  }
  for (const e of entries) {
    const hints = [...new Set(e.hints)];
    if (!e.errors.length && !hints.length) continue;
    if (e.errors.length) rejected++; else hinted++;
    typo ||= e.typo;
    console.log(e.errors.length ? `${e.name}　未收录` : e.name);
    for (const m of e.errors) console.log(`  错误　${m}`);
    for (const m of hints) console.log(`  提示　${m}`);
  }
  const parts = [`收录 ${tally(count)}`];
  if (enCount?.entries) parts.push(`英文版 ${tally(enCount)}`);
  if (rejected) parts.push(`未收录 ${rejected} 项`);
  if (hinted) parts.push(`${hinted} 项有提示`);
  console.log(`${rejected || hinted || tocHints.length ? '\n' : ''}${parts.join('，')}。${typo ? '排版问题可用 node build.js --fix 修正。' : ''}`);
  return rejected === 0;
}

function fix() {
  const changed = [];
  for (const [dir, f] of [WORDS, QUESTIONS, TOPICS].flatMap(dir => files(dir).map(f => [dir, f]))) {
    const src = read(f, dir), out = typeset(src);
    if (out !== src) {
      fs.writeFileSync(path.join(dir, f), `${out}\n`);
      changed.push(nameOf(f));
    }
  }
  if (changed.length) console.log(`已修正排版：${changed.join('、')}\n`);
}


let katex;
try {
  katex = require(KATEX);
} catch {
  console.error(`缺少 KaTeX。在项目目录中运行：\n\n  ${FETCH_KATEX}\n`);
  process.exit(1);
}

const args = process.argv.slice(2);
if (args.includes('--fix')) fix();

if (args.includes('-w')) {
  build();
  let timer;
  const rebuild = () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      console.log(`\n${new Date().toLocaleTimeString('zh-CN', { hour12: false })}`);
      build();
    }, 100);
  };
  for (const target of [WORDS, QUESTIONS, TOPICS, TOC, EN_WORDS, EN_QUESTIONS, EN_TOPICS]) if (fs.existsSync(target)) fs.watch(target, rebuild);
  console.log('正在监视 words/、questions/、topics/、en/ 与目录.md，按 Ctrl-C 退出。');
} else {
  process.exit(build() ? 0 : 1);
}
