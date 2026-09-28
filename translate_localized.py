#!/usr/bin/env python3
"""Перевод строк `string m_Localized = "..."` в дампах TableLocalizations (EN -> RU).

Как это работает
---------------
1. Из каждой строки вырезаются richText-теги (<i>, <size=60%>, <shake>, ...) и
   плейсхолдеры ({0}, {1} ...) — они заменяются на токены вида [[N]] и потом
   возвращаются на место, поэтому разметка не переводится и не теряется.
2. Строки группируются по «отпечатку» разметки (одинаковый набор тегов), и внутри
   группы переводятся батчами: несколько фраз отправляются одним запросом через
   переводчик Google (бесплатный эндпоинт translate.googleapis.com), что сильно
   экономит количество запросов.
3. Результаты кэшируются в .translation_cache.json — скрипт можно прерывать и
   перезапускать, переведённое повторно не запрашивается.

Использование:
    python3 translate_localized.py Dialogue.txt content.txt
"""

import json
import os
import re
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import requests

TARGET_LANG = "ru"
SOURCE_LANG = "en"
API_URL = "https://translate.googleapis.com/translate_a/single"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(HERE, ".translation_cache.json")
WORKERS = 4                    # параллельных потоков
MAX_CHARS_PER_REQUEST = 1200   # максимум символов в одном запросе
SEPARATOR = "\n"               # разделитель фраз внутри батча
SENT_SPLIT_RE = re.compile(r'(?<=[.!?…])\s+')

LINE_RE = re.compile(r'^(\s*\d+ string m_Localized = ")(.*)("\s*)$')
PROTECT_RE = re.compile(r'<[^<>]+>|\{\d+\}')
TOKEN_RE = re.compile(r'\[\[(\d+)\]\]')


# ------------------------------------------------------------------ токенизация
def protect(text):
    """Возвращает (текст с токенами [[N]], список защищённых фрагментов)."""
    protected = []

    def repl(m):
        protected.append(m.group(0))
        return f'[[{len(protected) - 1}]]'

    return PROTECT_RE.sub(repl, text), protected


def restore(translated, protected):
    """Ставит защищённые фрагменты обратно; потерянные дописывает в конец."""
    used = set()

    def sub(m):
        idx = int(m.group(1))
        if 0 <= idx < len(protected):
            used.add(idx)
            return protected[idx]
        return m.group(0)

    result = TOKEN_RE.sub(sub, translated)
    missing = [protected[i] for i in range(len(protected)) if i not in used]
    if missing:
        result += ' ' + ' '.join(missing)
    return result


def fingerprint(masked):
    """Отпечаток разметки: токены + небуквенные символы вокруг них."""
    return re.sub(r'[A-Za-z0-9]', '', masked)


def split_sentences(text, limit):
    """Режет длинный текст на куски <= limit по предложениям (токены не ломаются)."""
    if len(text) <= limit:
        return [text]
    parts, cur = [], ''
    for sent in SENT_SPLIT_RE.split(text):
        piece = sent + (' ' if not sent.endswith(' ') else '')
        if cur and len(cur) + len(piece) > limit:
            parts.append(cur.rstrip())
            cur = ''
        cur += piece
        while len(cur) > limit:                 # одно очень длинное предложение
            cut = cur.rfind(' ', 0, limit) or limit
            parts.append(cur[:cut].rstrip())
            cur = cur[cut:]
    if cur.strip():
        parts.append(cur.rstrip())
    return parts


# ------------------------------------------------------------------ переводчик
class Translator:
    def __init__(self):
        self.local = threading.local()
        self.lock = threading.Lock()
        self.cache = {'full': {}, 'batch': {}}
        self.writes = 0
        self.errors = []
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, encoding='utf-8') as fh:
                    data = json.load(fh)
                self.cache['full'].update(data.get('full', {}))
                self.cache['batch'].update(data.get('batch', {}))
            except Exception:
                pass

    def session(self):
        s = getattr(self.local, 'session', None)
        if s is None:
            s = requests.Session()
            s.headers['User-Agent'] = 'Mozilla/5.0'
            self.local.session = s
        return s

    # -- кэш ------------------------------------------------------------
    def save_cache(self):
        with self.lock:
            tmp = CACHE_FILE + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump(self.cache, fh, ensure_ascii=False)
            os.replace(tmp, CACHE_FILE)

    # -- HTTP -----------------------------------------------------------
    def _request(self, text):
        last_err = None
        for attempt in range(5):
            try:
                resp = self.session().get(
                    API_URL,
                    params={'client': 'gtx', 'sl': SOURCE_LANG, 'tl': TARGET_LANG,
                            'dt': 't', 'q': text},
                    timeout=30,
                )
                if resp.status_code == 429:
                    last_err = 'HTTP 429'
                    time.sleep(1.5 + attempt * 2.0)
                    continue
                resp.raise_for_status()
                data = resp.json()
                return ''.join(seg[0] for seg in data[0] if seg and seg[0])
            except Exception as exc:
                last_err = repr(exc)
                time.sleep(1.0 + attempt * 1.5)
        raise RuntimeError(f'перевод не удался [{last_err}]')

    def translate_batch(self, texts):
        """Переводит список фраз (без токенов); вернуть должен список того же размера."""
        results = [None] * len(texts)
        pending = []
        for i, t in enumerate(texts):
            cached = self.cache['batch'].get(t)
            if cached is not None:
                results[i] = cached
            else:
                pending.append(i)

        pos = 0
        while pos < len(pending):
            batch, size = [], 0
            while pos < len(pending) and (not batch or
                                          size + len(texts[pending[pos]]) + 1 <= MAX_CHARS_PER_REQUEST):
                j = pending[pos]
                batch.append(j)
                size += len(texts[j]) + 1
                pos += 1
            joined = SEPARATOR.join(texts[j] for j in batch)
            try:
                translated = self._request(joined)
            except Exception as exc:
                with self.lock:
                    self.errors.append((joined[:100], str(exc)))
                for j in batch:
                    results[j] = texts[j]      # оставляем оригинал
                continue
            parts = translated.split(SEPARATOR)
            if len(parts) != len(batch):
                parts = []
                for j in batch:
                    try:
                        parts.append(self._request(texts[j]))
                    except Exception as exc:
                        with self.lock:
                            self.errors.append((texts[j][:100], str(exc)))
                        parts.append(texts[j])
            for j, part in zip(batch, parts):
                results[j] = part
                with self.lock:
                    self.cache['batch'][texts[j]] = part
                    self.writes += 1
                    if self.writes >= 400:
                        self.writes = 0
                        self.save_cache()
        return results

    # -- одна строка текста с разметкой ---------------------------------
    def translate_line(self, text):
        key = text.strip()
        if not key:
            return text
        hit = self.cache['full'].get(key)
        if hit is not None:
            return hit
        masked, protected = protect(key)
        pieces = split_sentences(masked, MAX_CHARS_PER_REQUEST // 2)
        out = ' '.join(self.translate_batch(pieces)).strip()
        out = restore(out, protected)
        with self.lock:
            self.cache['full'][key] = out
        return out


# ------------------------------------------------------------------ сбор строк
def collect_lines(path):
    with open(path, encoding='utf-8') as fh:
        content = fh.read()
    trailing = content.endswith('\n')
    lines = content.split('\n')
    if trailing:
        lines.pop()
    return lines, trailing


def build_batches(strings, tr, max_chars=3000):
    """Группирует строки по отпечатку разметки и упаковывает в батчи для запросов."""
    groups = defaultdict(list)
    for s in strings:
        masked, _ = protect(s)
        groups[fingerprint(masked)].append(s)

    batches = []
    for members in groups.values():
        cur, cur_len = [], 0
        for s in members:
            extra = len(s) + 1
            if cur and cur_len + extra > max_chars:
                batches.append(cur)
                cur, cur_len = [], 0
            cur.append(s)
            cur_len += extra
        if cur:
            batches.append(cur)
    return batches


def run_translations(tr, todo, label=''):
    """Параллельно переводит список уникальных строк, пишет прогресс."""
    done = {'n': 0}
    total = len(todo)
    lock = threading.Lock()

    def work(chunk):
        for src in chunk:
            try:
                tr.translate_line(src)
            except Exception as exc:
                with tr.lock:
                    tr.errors.append((src[:100], str(exc)))
        with lock:
            done['n'] += len(chunk)
            if done['n'] % 300 < len(chunk) or done['n'] >= total:
                print(f'  {label}прогресс: {min(done["n"], total)}/{total}', flush=True)
                tr.save_cache()

    batches = build_batches(todo, tr)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(work, batches))
    tr.save_cache()
    return len(batches)


# ------------------------------------------------------------------ файл
def process_file(path, tr):
    lines, trailing = collect_lines(path)
    targets, originals = [], []
    for idx, line in enumerate(lines):
        m = LINE_RE.match(line)
        if m:
            targets.append(idx)
            originals.append(m.group(2))

    uniq = list(dict.fromkeys(s.strip() for s in originals if s.strip()))
    todo = [s for s in uniq if s not in tr.cache['full']]
    print(f'{path}: строк m_Localized = {len(originals)}, уникальных = {len(uniq)}, '
          f'новых для перевода = {len(todo)}', flush=True)

    nreq = 0
    if todo:
        nreq = run_translations(tr, todo, label=f'{os.path.basename(path)}: ')
        print(f'  запросов к переводчику: ~{nreq}', flush=True)

    changed = 0
    for idx, orig in zip(targets, originals):
        new = tr.cache['full'].get(orig.strip(), orig)
        new = new.replace('"', "'").replace('\n', ' ').strip()
        if new != orig:
            changed += 1
        m = LINE_RE.match(lines[idx])
        lines[idx] = m.group(1) + new + m.group(3)

    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
        if trailing:
            fh.write('\n')
    print(f'  готово: переведено {changed} из {len(originals)} строк -> {path}', flush=True)
    if tr.errors:
        print(f'  ошибок запроса на данный момент: {len(tr.errors)}', flush=True)


def main():
    files = sys.argv[1:] or ['Dialogue.txt', 'content.txt']
    tr = Translator()
    for f in files:
        process_file(f, tr)
    tr.save_cache()
    print('Кэш перевода:', CACHE_FILE)
    if tr.errors:
        print(f'ВНИМАНИЕ: {len(tr.errors)} фрагментов не переведены (остался английский). '
              f'Перезапустите скрипт, чтобы доперевести.')


if __name__ == '__main__':
    main()
