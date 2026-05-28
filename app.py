# -*- coding: utf-8 -*-
"""
Auditor Nómina Part Time JMC
Creado por Andrés Huérfano Dávila – Nómina JMC

Motor Streamlit Cloud para revisar nómina Part Time ZH/ZP:
- Base y pago de vacaciones con acumulados 365, AUSNOM acumulado e histórico de salarios.
- Comparación contra prenómina SAP por concepto: salario, bono antigüedad, vacaciones, auxilio, BigPass.
- Revisión de IBC / Seguridad Social: 9262, 9263, Z000, Z010 y referencia /662 opcional.
- Calendario interno Colombia.
"""

from __future__ import annotations

import io
import re
import zipfile
import unicodedata
import calendar
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import holidays
except Exception:
    holidays = None

# =============================
# Configuración
# =============================
APP_TITLE = "Auditor Nómina Part Time"
APP_ICON = "🦜"
DIAS_BASE = 30

MESES_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}
DIAS_ES = {0: "lunes", 1: "martes", 2: "miércoles", 3: "jueves", 4: "viernes", 5: "sábado", 6: "domingo"}

SALARY_ZH = "Y090"
SALARY_ZP = "Y011"
BONUS_ANT = "Y617"
VACATION_CODE = "Y400"
AUX_CODES_DEFAULT = ["Y200"]
BIGPASS_CODES_DEFAULT = ["Y598"]
BIGPASS_DED_CODES_DEFAULT = ["Z590"]
IBC_CODES_DEFAULT = ["9262", "9263"]
HEALTH_CODE = "Z000"
PENSION_CODE = "Z010"

NON_SALARY_DEFAULT = ["Y200", "Y598", "Z590"]
BASE_SALARY_CODES = [SALARY_ZH, SALARY_ZP, BONUS_ANT]

# =============================
# Utilidades básicas
# =============================
def norm_text(value) -> str:
    if pd.isna(value):
        return ""
    txt = str(value).strip().lower()
    txt = unicodedata.normalize("NFKD", txt)
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
    txt = re.sub(r"\s+", " ", txt)
    return txt


def norm_code(value) -> str:
    if pd.isna(value):
        return ""
    txt = str(value).strip().upper()
    txt = re.sub(r"\s+", "", txt)
    return txt


def make_unique_columns(cols: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out = []
    for c in cols:
        base = str(c).strip()
        if not base or base.lower() == "nan":
            base = "Columna"
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
    return out


def safe_num_series(s) -> pd.Series:
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    if s is None:
        return pd.Series(dtype=float)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0.0)
    ser = s.astype(str).str.strip()
    ser = ser.str.replace("$", "", regex=False).str.replace("COP", "", regex=False)
    ser = ser.str.replace("\u00a0", "", regex=False).str.replace(" ", "", regex=False)
    # Formato colombiano: 1.750.905 o 1.750.905,50. También tolera 1750905.
    ser = ser.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    ser = ser.replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan})
    return pd.to_numeric(ser, errors="coerce").fillna(0.0)


def safe_num_value(x) -> float:
    return float(safe_num_series(pd.Series([x])).iloc[0])


def safe_date_series(s) -> pd.Series:
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    if s is None:
        return pd.Series(dtype=object)
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce").dt.date
    out = pd.to_datetime(s, errors="coerce", dayfirst=True)
    mask = out.isna()
    if mask.any():
        nums = pd.to_numeric(s[mask], errors="coerce")
        serial = pd.to_datetime(nums, unit="D", origin="1899-12-30", errors="coerce")
        out.loc[mask] = serial
    return out.dt.date


def clean_employee_id(s) -> pd.Series:
    ser = s.astype(str).str.strip()
    ser = ser.str.replace(r"\.0$", "", regex=True)
    extracted = ser.str.extract(r"(\d+)", expand=False)
    return extracted.fillna(ser).str.strip()


def find_col(df: pd.DataFrame, candidates: Iterable[str], contains: bool = True) -> Optional[str]:
    norm_cols = {c: norm_text(c) for c in df.columns}
    cand = [norm_text(x) for x in candidates]
    for c, nc in norm_cols.items():
        if nc in cand:
            return c
    if contains:
        for c, nc in norm_cols.items():
            if any(k in nc for k in cand):
                return c
    return None


def sum_codes(df: pd.DataFrame, codes: Iterable[str], value_col: str = "Valor") -> pd.DataFrame:
    codes_set = {norm_code(c) for c in codes}
    if df.empty:
        return pd.DataFrame(columns=["SAP", value_col])
    tmp = df[df["Código"].isin(codes_set)].groupby("SAP", as_index=False)[value_col].sum()
    return tmp


def qty_codes(df: pd.DataFrame, codes: Iterable[str]) -> pd.DataFrame:
    codes_set = {norm_code(c) for c in codes}
    if df.empty:
        return pd.DataFrame(columns=["SAP", "Cantidad"])
    return df[df["Código"].isin(codes_set)].groupby("SAP", as_index=False)["Cantidad"].sum()


def compare_status(diff: float, tolerance: float) -> str:
    if pd.isna(diff):
        return "REVISAR"
    return "OK" if abs(float(diff)) <= float(tolerance) else "REVISAR"


def money_fmt(v) -> str:
    try:
        return f"${float(v):,.0f}".replace(",", ".")
    except Exception:
        return str(v)

# =============================
# Lectura de archivos
# =============================
def parse_pipe_txt(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    if isinstance(raw, bytes):
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                text = raw.decode(enc)
                break
            except Exception:
                text = raw.decode("latin-1", errors="ignore")
    else:
        text = str(raw)

    lines = text.splitlines()
    pipe_lines = [ln for ln in lines if "|" in ln]
    if not pipe_lines:
        raise ValueError("El TXT no parece ser una salida SAP con separadores pipe |.")

    # Busca una fila de encabezado con campos típicos.
    header_idx = 0
    for i, ln in enumerate(pipe_lines[:80]):
        n = norm_text(ln)
        if ("nº pers" in n or "n pers" in n or "numero" in n) and ("importe" in n or "valido" in n or "cc-n" in n):
            header_idx = i
            break

    header = [x.strip() for x in pipe_lines[header_idx].strip().strip("|").split("|")]
    header = make_unique_columns(header)
    rows = []
    for ln in pipe_lines[header_idx + 1:]:
        if set(ln.strip()) <= {"-", "|"}:
            continue
        parts = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(parts) != len(header):
            # tolera líneas raras del reporte.
            continue
        if not any(parts):
            continue
        rows.append(parts)
    return pd.DataFrame(rows, columns=header)


def detect_header_row(raw: pd.DataFrame, required_any: Iterable[str], max_rows: int = 40) -> int:
    keys = [norm_text(k) for k in required_any]
    best_row, best_score = 0, -1
    for idx in range(min(max_rows, len(raw))):
        joined = " | ".join(norm_text(x) for x in raw.iloc[idx].tolist())
        score = sum(1 for k in keys if k in joined)
        if score > best_score:
            best_row, best_score = idx, score
    return best_row


def read_excel_table(uploaded_file, sheet_name: Optional[str], kind: str) -> pd.DataFrame:
    uploaded_file.seek(0)
    raw = pd.read_excel(uploaded_file, sheet_name=sheet_name or 0, header=None, dtype=object)
    if kind == "preno":
        keys = ["codigo", "concepto", "valor", "sap", "cantidad"]
    elif kind == "md":
        keys = ["nº pers", "area de nomina", "cc-nomina", "importe"]
    elif kind == "conceptos":
        keys = ["cc-n", "texto", "concepto"]
    elif kind == "hist":
        keys = ["nº pers", "cc-nomina", "importe", "desde", "hasta"]
    elif kind == "ausnom":
        keys = ["nº pers", "txt.cl.pres", "valido de", "valido a", "area nom"]
    else:
        keys = ["nº pers", "cc-n", "importe"]
    header_row = detect_header_row(raw, keys)
    uploaded_file.seek(0)
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name or 0, header=header_row, dtype=object)
    df = df.dropna(how="all")
    df.columns = make_unique_columns([str(c).strip() for c in df.columns])
    return df


def read_any_table(uploaded_file, kind: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    name = getattr(uploaded_file, "name", "").lower()
    if name.endswith(".txt") or name.endswith(".csv"):
        return parse_pipe_txt(uploaded_file)
    return read_excel_table(uploaded_file, sheet_name, kind)


def list_sheets(uploaded_file) -> List[str]:
    name = getattr(uploaded_file, "name", "").lower()
    if not name.endswith((".xlsx", ".xls", ".xlsm")):
        return []
    uploaded_file.seek(0)
    try:
        return pd.ExcelFile(uploaded_file).sheet_names
    finally:
        uploaded_file.seek(0)

# =============================
# Calendario Colombia y reglas PT
# =============================
def fallback_colombia_holidays(year: int) -> Dict[date, str]:
    def easter(y: int) -> date:
        a = y % 19; b = y // 100; c = y % 100; d = b // 4; e = b % 4
        f = (b + 8) // 25; g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4; k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        return date(y, month, day)
    def next_monday(d: date) -> date:
        return d + timedelta(days=(7 - d.weekday()) % 7)
    e = easter(year)
    return {
        date(year, 1, 1): "Año Nuevo",
        next_monday(date(year, 1, 6)): "Reyes Magos",
        next_monday(date(year, 3, 19)): "San José",
        date(year, 5, 1): "Día del Trabajo",
        next_monday(date(year, 6, 29)): "San Pedro y San Pablo",
        date(year, 7, 20): "Independencia",
        date(year, 8, 7): "Batalla de Boyacá",
        next_monday(date(year, 8, 15)): "Asunción de la Virgen",
        next_monday(date(year, 10, 12)): "Día de la Raza",
        next_monday(date(year, 11, 1)): "Todos los Santos",
        next_monday(date(year, 11, 11)): "Independencia de Cartagena",
        date(year, 12, 8): "Inmaculada Concepción",
        date(year, 12, 25): "Navidad",
        e - timedelta(days=3): "Jueves Santo",
        e - timedelta(days=2): "Viernes Santo",
        next_monday(e + timedelta(days=39)): "Ascensión del Señor",
        next_monday(e + timedelta(days=60)): "Corpus Christi",
        next_monday(e + timedelta(days=68)): "Sagrado Corazón",
    }


def get_colombia_holidays(year: int) -> Dict[date, str]:
    if holidays is not None:
        try:
            co = holidays.Colombia(years=[year])
            return {d: str(name) for d, name in co.items()}
        except Exception:
            pass
    return fallback_colombia_holidays(year)


def is_holiday_co(d: date) -> bool:
    return d in get_colombia_holidays(d.year)


def build_calendar(year: int, month: int) -> pd.DataFrame:
    hols = get_colombia_holidays(year)
    rows = []
    for day in range(1, calendar.monthrange(year, month)[1] + 1):
        d = date(year, month, day)
        wd = d.weekday()
        rows.append({
            "Fecha": d,
            "Día semana": DIAS_ES[wd],
            "Es festivo Colombia": d in hols,
            "Festivo": hols.get(d, ""),
            "Es sábado": wd == 5,
            "Es domingo": wd == 6,
            "Es lunes festivo": wd == 0 and d in hols,
            "Cuenta ZH base L-V": wd <= 4,
            "Cuenta ZP base FDS": wd in (5, 6) or (wd == 0 and d in hols),
        })
    return pd.DataFrame(rows)


def area_group(area_text: str) -> str:
    t = norm_text(area_text)
    if t == "zh" or "tiempor parcial hora" in t or "tiempo parcial hora" in t or "part time horas" in t or "part-time horas" in t:
        return "ZH"
    if t == "zp" or "tiempo parcial dia" in t or "parcial dia" in t or "fds" in t or "part time fds" in t:
        return "ZP"
    return "NO_IDENTIFICADA"


def absence_family(abs_text: str) -> str:
    t = norm_text(abs_text)
    if "vacacion" in t or "vacaciones" in t:
        return "VACACIONES"
    if "ausencia no just" in t or "no justificada" in t or t.startswith("anj") or "sin soporte" in t:
        return "ANJ"
    if "licencia no remunerada" in t or "lic no remuner" in t or "unpaid" in t:
        return "LNR"
    if "suspension" in t or "suspensión" in str(abs_text).lower():
        return "SUSPENSION"
    if "incap" in t or "inca" in t or "prorroga" in t or "enfermedad" in t or "accid" in t:
        return "INCAPACIDAD"
    if "dia de la familia" in t or "calamidad" in t or "luto" in t or "licencia remunerada" in t:
        return "REMUNERADA"
    return "OTRA"


def dates_between(start: date, end: date) -> List[date]:
    if not start or not end or pd.isna(start) or pd.isna(end) or end < start:
        return []
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def include_day_by_rule(family: str, area: str, d: date) -> bool:
    wd = d.weekday()
    holiday = is_holiday_co(d)
    if area == "ZH":
        if family in ["VACACIONES", "LNR", "SUSPENSION"]:
            return wd <= 4 and not holiday
        # ZH labora L-V, incluso si el día es festivo para ANJ/incap/remuneradas.
        return wd <= 4
    if area == "ZP":
        if family == "VACACIONES":
            return wd <= 5 and not holiday  # lunes a sábado, sin domingos ni festivos
        if family in ["LNR", "ANJ", "INCAPACIDAD", "REMUNERADA", "OTRA"]:
            return wd in (5, 6) or (wd == 0 and holiday)
        if family == "SUSPENSION":
            return wd == 5
    return False


def count_days_for_rule(family: str, area: str, start: date, end: date, clip_start: Optional[date] = None, clip_end: Optional[date] = None) -> Tuple[float, str]:
    if not start or not end:
        return 0.0, ""
    s = max(start, clip_start) if clip_start else start
    e = min(end, clip_end) if clip_end else end
    selected = [d for d in dates_between(s, e) if include_day_by_rule(family, area, d)]
    detail = ", ".join([f"{d.strftime('%d/%m/%Y')} {DIAS_ES[d.weekday()]}" for d in selected])
    return float(len(selected)), detail


def jornada_mensual_vigente(d: date) -> int:
    # Reducción jornada Colombia, usada por JMC para valor hora PT.
    if d < date(2025, 7, 15):
        return 230
    if d < date(2026, 7, 15):
        return 220
    return 210

# =============================
# Preparación de fuentes
# =============================
def prepare_md(df: pd.DataFrame) -> pd.DataFrame:
    sap_col = find_col(df, ["Nº pers.", "N° pers.", "Numero de personal", "Número de personal", "SAP"], contains=True)
    name_col = find_col(df, ["Número de personal", "Nombre", "Nombre del empleado"], contains=True)
    area_col = find_col(df, ["Área de nómina", "Texto área nómina", "Area de nomina", "ÁreaCálcNómPP"], contains=True)
    salary_col = find_col(df, ["Importe", "Salario", "Sueldo"], contains=True)
    ceco_col = find_col(df, ["Ce.coste", "Centro de coste", "Centro de costo", "CECO"], contains=True)
    cargo_col = find_col(df, ["Cargo", "Función", "Posición"], contains=True)
    ced_col = find_col(df, ["Número ID", "Cedula", "Cédula"], contains=True)
    if not sap_col or not area_col:
        raise ValueError("No se pudo identificar en MD las columnas SAP/Nº pers. y Área de nómina.")
    out = pd.DataFrame()
    out["SAP"] = clean_employee_id(df[sap_col])
    out["Nombre"] = df[name_col].astype(str) if name_col else ""
    out["Cédula"] = clean_employee_id(df[ced_col]) if ced_col else ""
    out["Área nómina"] = df[area_col].astype(str)
    out["Grupo área"] = out["Área nómina"].apply(area_group)
    out["Salario MD"] = safe_num_series(df[salary_col]) if salary_col else 0.0
    out["CECO"] = df[ceco_col].astype(str) if ceco_col else ""
    out["Cargo"] = df[cargo_col].astype(str) if cargo_col else ""
    out = out[out["SAP"].str.len() > 0].drop_duplicates("SAP", keep="last")
    out = out[out["Grupo área"].isin(["ZH", "ZP"])].copy()
    return out


def prepare_preno(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    code_col = find_col(df, ["CÓDIGO", "CODIGO", "Código", "CC-n.", "CC-nómina", "CC nomina"], contains=True)
    concept_col = find_col(df, ["CONCEPTO", "Texto expl.CC-nómina", "Txt.explicativo", "Concepto"], contains=True)
    qty_col = find_col(df, ["CANTIDAD", "CANT", "Cantidad", "Día/hora", "Ctd."], contains=True)
    val_col = find_col(df, ["VALOR", "Importe", "Valor"], contains=True)
    sap_col = find_col(df, ["SAP", "Nº pers.", "Número de personal", "Numero de personal"], contains=True)
    name_col = find_col(df, ["NOMBRE", "Nombre", "Nombre del empleado"], contains=True)
    salary_col = find_col(df, ["SALARIO", "Sueldo Básico", "Sueldo"], contains=True)
    ced_col = find_col(df, ["CÉDULA", "CEDULA", "Número ID"], contains=True)
    if not sap_col or not code_col or not val_col:
        raise ValueError("No se pudo identificar en Prenómina las columnas SAP, Código y Valor.")
    out = pd.DataFrame()
    out["SAP"] = clean_employee_id(df[sap_col])
    out["Código"] = df[code_col].astype(str).map(norm_code)
    out["Concepto"] = df[concept_col].astype(str) if concept_col else ""
    out["Cantidad"] = safe_num_series(df[qty_col]) if qty_col else 0.0
    out["Valor"] = safe_num_series(df[val_col])
    out["Nombre Preno"] = df[name_col].astype(str) if name_col else ""
    out["Salario recibo"] = safe_num_series(df[salary_col]) if salary_col else 0.0
    out["Cédula"] = clean_employee_id(df[ced_col]) if ced_col else ""
    out = out[(out["SAP"].str.len() > 0) & (out["Código"].str.len() > 0)].copy()

    net_df = pd.DataFrame()
    # Si la hoja trae neto mezclado, lo detecta. Normalmente viene en hoja NETOS y se lee aparte si el usuario la carga en el mismo Excel.
    return out, net_df


def prepare_acum(df: pd.DataFrame) -> pd.DataFrame:
    sap_col = find_col(df, ["Nº pers.", "N° pers.", "Número de personal", "Numero de personal", "SAP"], contains=True)
    date_col = find_col(df, ["Fecha pago", "Fecha de pago", "Per.para", "Período"], contains=True)
    code_col = find_col(df, ["CC-n.", "CC-nómina", "CC nomina"], contains=True)
    concept_col = find_col(df, ["Texto expl.CC-nómina", "Txt.explicativo", "Concepto"], contains=True)
    qty_col = find_col(df, ["Cantidad", "CANT", "Ctd."], contains=True)
    val_col = find_col(df, ["Importe", "Valor"], contains=True)
    if not sap_col or not date_col or not code_col or not val_col:
        raise ValueError("No se pudo identificar en Acumulados: SAP, Fecha pago, CC-n. e Importe.")
    out = pd.DataFrame()
    out["SAP"] = clean_employee_id(df[sap_col])
    out["Fecha pago"] = safe_date_series(df[date_col])
    out["Código"] = df[code_col].astype(str).map(norm_code)
    out["Concepto"] = df[concept_col].astype(str) if concept_col else ""
    out["Cantidad"] = safe_num_series(df[qty_col]) if qty_col else 0.0
    out["Valor"] = safe_num_series(df[val_col])
    return out.dropna(subset=["Fecha pago"])


def prepare_ausnom(df: pd.DataFrame, md: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    sap_col = find_col(df, ["Nº pers.", "Nº personal", "Número de personal", "Numero de personal", "SAP"], contains=True)
    name_col = find_col(df, ["Nom.empl", "Nombre", "Nombre del empleado", "candidato"], contains=True)
    abs_col = find_col(df, ["Txt.cl.pres./ab.", "Texto", "Ausentismo", "Clase absent", "cl.pres"], contains=True)
    start_col = find_col(df, ["Válido de", "Valido de", "Inicio de validez", "Fecha inicio", "Desde"], contains=True)
    end_col = find_col(df, ["Válido a", "Valido a", "Fin de validez", "Fecha fin", "Hasta"], contains=True)
    days_col = find_col(df, ["D.ab./pr.", "Días presenc./abs.", "Dias presenc", "Días", "Dias"], contains=True)
    area_col = find_col(df, ["Área nóm.", "Area nom", "Área de nómina", "Area de nomina"], contains=True)
    if not sap_col or not abs_col or not start_col or not end_col:
        raise ValueError("No se pudo identificar en AUSNOM: SAP, Ausentismo, Fecha inicio y Fecha fin.")
    out = pd.DataFrame()
    out["SAP"] = clean_employee_id(df[sap_col])
    out["Nombre Ausnom"] = df[name_col].astype(str) if name_col else ""
    out["Ausentismo SAP"] = df[abs_col].astype(str)
    out["Familia ausencia"] = out["Ausentismo SAP"].apply(absence_family)
    out["Fecha inicio"] = safe_date_series(df[start_col])
    out["Fecha fin"] = safe_date_series(df[end_col])
    out["Días SAP"] = safe_num_series(df[days_col]) if days_col else 0.0
    out["Área AUSNOM"] = df[area_col].astype(str) if area_col else ""
    out["Grupo área"] = out["Área AUSNOM"].apply(area_group)
    if md is not None and not md.empty:
        out = out.merge(md[["SAP", "Grupo área", "Nombre"]].rename(columns={"Grupo área": "Grupo área MD", "Nombre": "Nombre MD"}), on="SAP", how="left")
        out["Grupo área"] = np.where(out["Grupo área"].isin(["ZH", "ZP"]), out["Grupo área"], out["Grupo área MD"])
        out["Nombre Ausnom"] = np.where(out["Nombre Ausnom"].astype(str).str.len() > 0, out["Nombre Ausnom"], out["Nombre MD"].fillna(""))
        out = out.drop(columns=[c for c in ["Grupo área MD", "Nombre MD"] if c in out.columns])
    out["Grupo área"] = out["Grupo área"].fillna("NO_IDENTIFICADA")
    return out.dropna(subset=["Fecha inicio", "Fecha fin"])


def prepare_hist(df: pd.DataFrame) -> pd.DataFrame:
    sap_col = find_col(df, ["Nº pers.", "N° pers.", "Número de personal", "Numero de personal", "SAP"], contains=True)
    name_col = find_col(df, ["Número de personal", "Nombre"], contains=True)
    area_col = find_col(df, ["Área de nómina", "Area de nomina"], contains=True)
    code_col = find_col(df, ["CC-nómina", "CC nomina", "CC-n."], contains=True)
    val_col = find_col(df, ["Importe", "Valor"], contains=True)
    # Hay dos columnas Desde. En el histórico, la fecha de vigencia va después de Mon. y antes de Hasta.
    hasta_col = find_col(df, ["Hasta"], contains=True)
    desde_candidates = [c for c in df.columns if norm_text(c).startswith("desde")]
    if len(desde_candidates) >= 2:
        vig_desde_col = desde_candidates[-1]
    else:
        vig_desde_col = desde_candidates[0] if desde_candidates else None
    if not sap_col or not code_col or not val_col or not vig_desde_col or not hasta_col:
        raise ValueError("No se pudo identificar en Histórico salarios: SAP, CC-nómina, Importe, Desde vigencia y Hasta.")
    out = pd.DataFrame()
    out["SAP"] = clean_employee_id(df[sap_col])
    out["Nombre Hist"] = df[name_col].astype(str) if name_col else ""
    out["Área nómina Hist"] = df[area_col].astype(str) if area_col else ""
    out["Grupo área Hist"] = out["Área nómina Hist"].apply(area_group)
    out["Código"] = df[code_col].astype(str).map(norm_code)
    out["Importe vigente"] = safe_num_series(df[val_col])
    out["Desde vigencia"] = safe_date_series(df[vig_desde_col])
    # 31.12.9999 puede fallar; lo tratamos manual.
    hasta_raw = df[hasta_col].astype(str).str.strip()
    hasta = safe_date_series(df[hasta_col])
    hasta = pd.Series([date(2099, 12, 31) if "9999" in str(x) else y for x, y in zip(hasta_raw, hasta)])
    out["Hasta vigencia"] = hasta
    return out.dropna(subset=["Desde vigencia", "Hasta vigencia"])


def prepare_662(df: pd.DataFrame) -> pd.DataFrame:
    sap_col = find_col(df, ["Nº pers.", "N° pers.", "Número de personal", "Numero de personal", "SAP"], contains=True)
    code_col = find_col(df, ["CC-n.", "CC-nómina", "CC nomina"], contains=True)
    val_col = find_col(df, ["Importe", "Valor"], contains=True)
    qty_col = find_col(df, ["Cantidad", "CANT", "Ctd."], contains=True)
    date_col = find_col(df, ["Fecha pago", "Per.para", "Periodo"], contains=True)
    if not sap_col or not code_col or not val_col:
        raise ValueError("No se pudo identificar en /662: SAP, CC-n. e Importe.")
    out = pd.DataFrame()
    out["SAP"] = clean_employee_id(df[sap_col])
    out["Código"] = df[code_col].astype(str).map(norm_code)
    out["Cantidad"] = safe_num_series(df[qty_col]) if qty_col else 0.0
    out["Valor /662"] = safe_num_series(df[val_col])
    out["Fecha pago /662"] = safe_date_series(df[date_col]) if date_col else pd.NaT
    out = out[out["Código"].eq("/662")]
    return out.groupby("SAP", as_index=False).agg({"Valor /662": "sum", "Cantidad": "sum"})


def prepare_variable_codes(df: pd.DataFrame) -> List[str]:
    code_col = find_col(df, ["CC-n.", "CC-nómina", "Código", "Codigo"], contains=True)
    if not code_col:
        raise ValueError("No se encontró columna de código en matriz de conceptos base variable.")
    codes = sorted({norm_code(x) for x in df[code_col].dropna().tolist() if norm_code(x)})
    return codes

# =============================
# Históricos y cálculo salario
# =============================
def hist_amount_for_date(hist: pd.DataFrame, sap: str, code: str, d: date) -> float:
    if hist is None or hist.empty or not d:
        return 0.0
    tmp = hist[(hist["SAP"] == str(sap)) & (hist["Código"] == norm_code(code))]
    tmp = tmp[(tmp["Desde vigencia"] <= d) & (tmp["Hasta vigencia"] >= d)]
    if tmp.empty:
        # fallback último anterior
        tmp = hist[(hist["SAP"] == str(sap)) & (hist["Código"] == norm_code(code)) & (hist["Desde vigencia"] <= d)]
        if tmp.empty:
            return 0.0
        tmp = tmp.sort_values("Desde vigencia")
    return float(tmp.iloc[-1]["Importe vigente"])


def area_for_sap(md: pd.DataFrame, sap: str) -> str:
    if md is None or md.empty:
        return "NO_IDENTIFICADA"
    tmp = md[md["SAP"] == str(sap)]
    if tmp.empty:
        return "NO_IDENTIFICADA"
    return str(tmp.iloc[0]["Grupo área"])


def salary_parts_for_date(hist: pd.DataFrame, sap: str, area: str, d: date) -> Dict[str, float]:
    base_code = SALARY_ZH if area == "ZH" else SALARY_ZP if area == "ZP" else ""
    base = hist_amount_for_date(hist, sap, base_code, d) if base_code else 0.0
    bonus = hist_amount_for_date(hist, sap, BONUS_ANT, d)
    return {
        "Código salario": base_code,
        "Salario base vigente": base,
        "Bono antigüedad vigente": bonus,
        "Salario total vigente": base + bonus,
        "Jornada vigente": jornada_mensual_vigente(d),
    }


def daily_value_for_absence(hist: pd.DataFrame, sap: str, area: str, d: date) -> Tuple[float, str, float]:
    parts = salary_parts_for_date(hist, sap, area, d)
    total = parts["Salario total vigente"]
    if total <= 0:
        return 0.0, "Sin histórico salarial", 0.0
    if area == "ZH":
        return total / parts["Jornada vigente"] * 4, "Histórico salarios PT", total
    if area == "ZP":
        return total / 30, "Histórico salarios PT", total
    return 0.0, "Área no identificada", total

# =============================
# Módulo vacaciones
# =============================
def build_vacation_module(
    acum: pd.DataFrame,
    aus: pd.DataFrame,
    md: pd.DataFrame,
    hist: pd.DataFrame,
    preno: pd.DataFrame,
    variable_codes: List[str],
    year: int,
    month: int,
    tolerance_money: float,
    absence_families_base: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    period_start = date(year, month, 1)
    period_end = date(year, month, calendar.monthrange(year, month)[1])

    vacs = aus[(aus["Familia ausencia"] == "VACACIONES") & (aus["Fecha inicio"] <= period_end) & (aus["Fecha fin"] >= period_start)].copy()
    rows = []
    dev_rows = []
    abs_rows = []
    base_codes = sorted(set(BASE_SALARY_CODES + variable_codes))
    preno_vac = sum_codes(preno, [VACATION_CODE], "Valor").rename(columns={"Valor": "Vacaciones SAP"})

    for _, vac in vacs.iterrows():
        sap = str(vac["SAP"])
        area = vac["Grupo área"] if vac["Grupo área"] in ["ZH", "ZP"] else area_for_sap(md, sap)
        start_vac = vac["Fecha inicio"]
        end_vac = vac["Fecha fin"]
        base_start = start_vac - timedelta(days=365)
        base_end = start_vac - timedelta(days=1)
        dias_vac_mes, detalle_vac = count_days_for_rule("VACACIONES", area, start_vac, end_vac, period_start, period_end)

        dev = acum[(acum["SAP"] == sap) & (acum["Fecha pago"] >= base_start) & (acum["Fecha pago"] <= base_end) & (acum["Código"].isin(base_codes))].copy()
        total_dev = float(dev["Valor"].sum()) if not dev.empty else 0.0
        if not dev.empty:
            tmp = dev.groupby(["SAP", "Fecha pago", "Código", "Concepto"], as_index=False).agg({"Cantidad": "sum", "Valor": "sum"})
            tmp["Ventana desde"] = base_start
            tmp["Ventana hasta"] = base_end
            tmp["Vacación inicio"] = start_vac
            dev_rows.append(tmp)

        aus_win = aus[(aus["SAP"] == sap) & (aus["Fecha inicio"] <= base_end) & (aus["Fecha fin"] >= base_start) & (aus["Familia ausencia"].isin(absence_families_base))].copy()
        total_abs = 0.0
        for _, a in aus_win.iterrows():
            s = max(a["Fecha inicio"], base_start)
            e = min(a["Fecha fin"], base_end)
            days, detail = count_days_for_rule(a["Familia ausencia"], area, s, e)
            # Valoriza por fecha de inicio real dentro de la ventana.
            val_dia, fuente, salario_usado = daily_value_for_absence(hist, sap, area, s)
            val_abs = days * val_dia
            total_abs += val_abs
            abs_rows.append(pd.DataFrame([{
                "SAP": sap,
                "Área": area,
                "Ausentismo": a["Ausentismo SAP"],
                "Familia": a["Familia ausencia"],
                "Fecha inicio": a["Fecha inicio"],
                "Fecha fin": a["Fecha fin"],
                "Tramo base desde": s,
                "Tramo base hasta": e,
                "Días reales promedio": days,
                "Detalle días": detail,
                "Salario usado": salario_usado,
                "Valor día usado": val_dia,
                "Valor a promedio": val_abs,
                "Fuente salario": fuente,
                "Vacación inicio": start_vac,
            }]))

        base_365 = total_dev + total_abs
        valor_dia_vac = base_365 / 365 if base_365 else 0.0
        esperado = valor_dia_vac * dias_vac_mes
        sap_paid = float(preno_vac.loc[preno_vac["SAP"] == sap, "Vacaciones SAP"].sum()) if not preno_vac.empty else 0.0
        diff = sap_paid - esperado
        obs = []
        if base_365 <= 0:
            obs.append("No se pudo calcular base 365: sin devengos ni ausentismos base en ventana")
        if sap_paid == 0:
            obs.append("No se encontró Y400 Vacaciones en prenómina para este empleado")
        if abs(diff) > tolerance_money:
            obs.append("Diferencia entre vacaciones SAP y cálculo esperado")
        if start_vac < period_start or end_vac > period_end:
            obs.append("Vacación cruza mes; se pagó/calculó solo tramo del mes revisado")

        rows.append({
            "SAP": sap,
            "Nombre": vac.get("Nombre Ausnom", ""),
            "Área": area,
            "Inicio vacaciones": start_vac,
            "Fin vacaciones": end_vac,
            "Días SAP AUSNOM": vac.get("Días SAP", 0.0),
            "Días vacaciones mes calculados": dias_vac_mes,
            "Detalle días vacaciones": detalle_vac,
            "Ventana promedio desde": base_start,
            "Ventana promedio hasta": base_end,
            "Devengos base acumulados": total_dev,
            "Ausentismos valorizados base": total_abs,
            "Base 365": base_365,
            "Valor diario vacaciones": valor_dia_vac,
            "Vacaciones esperadas": esperado,
            "Vacaciones SAP Y400": sap_paid,
            "Diferencia vacaciones": diff,
            "Estado": compare_status(diff, tolerance_money),
            "Observación": "; ".join(obs) if obs else "OK",
        })

    base_vac = pd.DataFrame(rows)
    dev_detail = pd.concat(dev_rows, ignore_index=True) if dev_rows else pd.DataFrame(columns=["SAP", "Fecha pago", "Código", "Concepto", "Cantidad", "Valor"])
    abs_detail = pd.concat(abs_rows, ignore_index=True) if abs_rows else pd.DataFrame(columns=["SAP", "Área", "Ausentismo", "Familia", "Días reales promedio", "Valor a promedio"])
    return base_vac, dev_detail, abs_detail

# =============================
# Módulo revisión mes
# =============================
def build_absence_month(aus: pd.DataFrame, year: int, month: int, tolerance_days: float) -> pd.DataFrame:
    period_start = date(year, month, 1)
    period_end = date(year, month, calendar.monthrange(year, month)[1])
    out = aus[(aus["Fecha inicio"] <= period_end) & (aus["Fecha fin"] >= period_start)].copy()
    if out.empty:
        return out
    calc = out.apply(lambda r: count_days_for_rule(r["Familia ausencia"], r["Grupo área"], r["Fecha inicio"], r["Fecha fin"], period_start, period_end), axis=1)
    out["Días calculados app"] = [x[0] for x in calc]
    out["Detalle días calculados"] = [x[1] for x in calc]
    out["Diferencia días SAP vs app"] = out["Días SAP"] - out["Días calculados app"]
    out["Estado"] = out["Diferencia días SAP vs app"].apply(lambda x: compare_status(x, tolerance_days))
    out["Observación"] = np.where(out["Estado"].eq("OK"), "OK", "Diferencia en días de ausentismo SAP vs regla Part Time")
    return out


def build_salary_review(preno: pd.DataFrame, md: pd.DataFrame, hist: pd.DataFrame, year: int, month: int, tolerance_money: float) -> pd.DataFrame:
    period_date = date(year, month, calendar.monthrange(year, month)[1])
    rows = []
    pt_saps = set(md["SAP"])
    pr = preno[preno["SAP"].isin(pt_saps)].copy()
    for _, emp in md.iterrows():
        sap = emp["SAP"]
        area = emp["Grupo área"]
        parts = salary_parts_for_date(hist, sap, area, period_date)
        base_code = parts["Código salario"]
        if not base_code:
            continue
        emp_pr = pr[pr["SAP"] == sap]
        sal_qty = float(emp_pr.loc[emp_pr["Código"] == base_code, "Cantidad"].sum())
        sal_val = float(emp_pr.loc[emp_pr["Código"] == base_code, "Valor"].sum())
        bonus_qty = float(emp_pr.loc[emp_pr["Código"] == BONUS_ANT, "Cantidad"].sum())
        bonus_val = float(emp_pr.loc[emp_pr["Código"] == BONUS_ANT, "Valor"].sum())
        if area == "ZH":
            expected_sal = parts["Salario base vigente"] / parts["Jornada vigente"] * sal_qty if parts["Jornada vigente"] else 0.0
            expected_bonus = parts["Bono antigüedad vigente"] / parts["Jornada vigente"] * bonus_qty if parts["Bono antigüedad vigente"] > 0 else 0.0
            unit = "horas"
        else:
            expected_sal = parts["Salario base vigente"] / 30 * sal_qty
            expected_bonus = parts["Bono antigüedad vigente"] / 30 * bonus_qty if parts["Bono antigüedad vigente"] > 0 else 0.0
            unit = "días"
        for concepto, cod, qty, sap_val, expected in [
            ("Salario base", base_code, sal_qty, sal_val, expected_sal),
            ("Bono antigüedad", BONUS_ANT, bonus_qty, bonus_val, expected_bonus),
        ]:
            diff = sap_val - expected
            obs = "OK"
            if cod == BONUS_ANT and parts["Bono antigüedad vigente"] <= 0 and sap_val == 0:
                obs = "Empleado sin bono antigüedad vigente"
            elif abs(diff) > tolerance_money:
                obs = f"Diferencia en {concepto}. Se calculó con histórico salarial vigente y cantidad SAP en {unit}."
            rows.append({
                "SAP": sap,
                "Nombre": emp.get("Nombre", ""),
                "Área": area,
                "Concepto revisión": concepto,
                "Código": cod,
                "Cantidad SAP": qty,
                "Valor SAP": sap_val,
                "Valor esperado": expected,
                "Diferencia": diff,
                "Salario base vigente": parts["Salario base vigente"],
                "Bono antigüedad vigente": parts["Bono antigüedad vigente"],
                "Jornada vigente": parts["Jornada vigente"],
                "Estado": compare_status(diff, tolerance_money),
                "Observación": obs,
            })
    return pd.DataFrame(rows)


def build_constants_review(preno: pd.DataFrame, md: pd.DataFrame, aux_value: float, bigpass_value: float, tolerance_money: float, aux_codes: List[str], bp_codes: List[str], bp_ded_codes: List[str]) -> pd.DataFrame:
    rows = []
    pt_saps = set(md["SAP"])
    pr = preno[preno["SAP"].isin(pt_saps)].copy()
    for _, emp in md.iterrows():
        sap = emp["SAP"]
        area = emp["Grupo área"]
        emp_pr = pr[pr["SAP"] == sap]
        if area == "ZH":
            days_base = float(emp_pr.loc[emp_pr["Código"] == SALARY_ZH, "Cantidad"].sum()) / 4
        else:
            days_base = float(emp_pr.loc[emp_pr["Código"] == SALARY_ZP, "Cantidad"].sum())
        aux_sap = float(emp_pr.loc[emp_pr["Código"].isin({norm_code(x) for x in aux_codes}), "Valor"].sum())
        bp_sap = float(emp_pr.loc[emp_pr["Código"].isin({norm_code(x) for x in bp_codes}), "Valor"].sum())
        bpd_sap = float(emp_pr.loc[emp_pr["Código"].isin({norm_code(x) for x in bp_ded_codes}), "Valor"].sum())
        for name, sap_val, expected in [
            ("Auxilio transporte", aux_sap, aux_value / 30 * days_base),
            ("BigPass ingreso", bp_sap, bigpass_value / 30 * days_base),
            ("BigPass descuento", bpd_sap, bigpass_value / 30 * days_base),
        ]:
            diff = sap_val - expected
            rows.append({
                "SAP": sap,
                "Nombre": emp.get("Nombre", ""),
                "Área": area,
                "Concepto revisión": name,
                "Días base usados": days_base,
                "Valor mensual parámetro": aux_value if "Auxilio" in name else bigpass_value,
                "Valor SAP": sap_val,
                "Valor esperado": expected,
                "Diferencia": diff,
                "Estado": compare_status(diff, tolerance_money),
                "Observación": "OK" if abs(diff) <= tolerance_money else f"Diferencia en {name}; días base tomados desde cantidad de salario SAP",
            })
    return pd.DataFrame(rows)


def build_ibc_ss_review(preno: pd.DataFrame, md: pd.DataFrame, aus_month: pd.DataFrame, base662: pd.DataFrame, variable_codes: List[str], excluded_codes: List[str], health_rate: float, pension_rate: float, tolerance_money: float) -> pd.DataFrame:
    pt_saps = set(md["SAP"])
    pr = preno[preno["SAP"].isin(pt_saps)].copy()
    excluded = {norm_code(c) for c in excluded_codes}
    # Incluye todos los Y* excepto aux/bp y variables salariales explícitas. Incluye vacaciones Y400 y Y1xx/Y2xx permisos.
    pr["Es salarial app"] = pr["Código"].apply(lambda c: (str(c).startswith("Y") and c not in excluded) or c in {norm_code(x) for x in variable_codes} or c in {VACATION_CODE})
    ibc_calc = pr[pr["Es salarial app"]].groupby("SAP", as_index=False)["Valor"].sum().rename(columns={"Valor": "IBC calculado app"})
    sap_9262 = sum_codes(pr, ["9262"], "Valor").rename(columns={"Valor": "SAP 9262"})
    sap_9263 = sum_codes(pr, ["9263"], "Valor").rename(columns={"Valor": "SAP 9263"})
    z000 = sum_codes(pr, [HEALTH_CODE], "Valor").rename(columns={"Valor": "Salud SAP Z000"})
    z010 = sum_codes(pr, [PENSION_CODE], "Valor").rename(columns={"Valor": "Pensión SAP Z010"})
    inc_days = pd.DataFrame(columns=["SAP", "Días incapacidad mes"])
    if aus_month is not None and not aus_month.empty:
        inc_days = aus_month[aus_month["Familia ausencia"].eq("INCAPACIDAD")].groupby("SAP", as_index=False)["Días calculados app"].sum().rename(columns={"Días calculados app": "Días incapacidad mes"})
    out = md[["SAP", "Nombre", "Grupo área"]].copy().rename(columns={"Grupo área": "Área"})
    for df in [ibc_calc, sap_9262, sap_9263, z000, z010, base662, inc_days]:
        if df is not None and not df.empty:
            out = out.merge(df, on="SAP", how="left")
    for c in ["IBC calculado app", "SAP 9262", "SAP 9263", "Salud SAP Z000", "Pensión SAP Z010", "Valor /662", "Días incapacidad mes"]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = out[c].fillna(0.0)
    out["Salud esperada app"] = (out["IBC calculado app"] * health_rate).round(0)
    out["Pensión esperada app"] = (out["IBC calculado app"] * pension_rate).round(0)
    out["Diferencia 9262"] = out["SAP 9262"] - out["IBC calculado app"]
    out["Diferencia 9263"] = out["SAP 9263"] - out["IBC calculado app"]
    out["Diferencia Salud"] = out["Salud SAP Z000"] - out["Salud esperada app"]
    out["Diferencia Pensión"] = out["Pensión SAP Z010"] - out["Pensión esperada app"]
    out["Estado"] = out.apply(lambda r: "OK" if all(abs(r[x]) <= tolerance_money for x in ["Diferencia Salud", "Diferencia Pensión"]) else "REVISAR", axis=1)
    def obs(r):
        msgs = []
        if abs(r["Diferencia Salud"]) > tolerance_money:
            msgs.append("Diferencia en salud Z000")
        if abs(r["Diferencia Pensión"]) > tolerance_money:
            msgs.append("Diferencia en pensión Z010")
        if abs(r["Diferencia 9262"]) > tolerance_money:
            msgs.append("IBC app difiere de 9262 SAP")
        if r["Días incapacidad mes"] > 0 and r["Valor /662"] <= 0:
            msgs.append("Tiene incapacidad y no se cargó/encontró /662 del mes anterior")
        if r["Días incapacidad mes"] > 0 and r["Valor /662"] > 0:
            msgs.append("Tiene incapacidad; /662 disponible como referencia para validación fina")
        return "; ".join(msgs) if msgs else "OK"
    out["Observación"] = out.apply(obs, axis=1)
    return out

# =============================
# Exportación
# =============================
def build_summary(md, preno, acum, aus, aus_month, base_vac, salary_rev, const_rev, ss_rev, calendar_df, logs) -> pd.DataFrame:
    return pd.DataFrame([
        {"Indicador": "Empleados Part Time evaluados", "Valor": len(md)},
        {"Indicador": "Empleados ZH", "Valor": int((md["Grupo área"] == "ZH").sum())},
        {"Indicador": "Empleados ZP", "Valor": int((md["Grupo área"] == "ZP").sum())},
        {"Indicador": "Registros prenómina", "Valor": len(preno)},
        {"Indicador": "Registros acumulados promedio", "Valor": len(acum)},
        {"Indicador": "Registros AUSNOM acumulado", "Valor": len(aus)},
        {"Indicador": "Ausentismos del mes a revisar", "Valor": int((aus_month.get("Estado", pd.Series(dtype=str)) == "REVISAR").sum()) if not aus_month.empty else 0},
        {"Indicador": "Vacaciones detectadas en el mes", "Valor": len(base_vac)},
        {"Indicador": "Vacaciones con diferencia", "Valor": int((base_vac.get("Estado", pd.Series(dtype=str)) == "REVISAR").sum()) if not base_vac.empty else 0},
        {"Indicador": "Salario / bono con diferencia", "Valor": int((salary_rev.get("Estado", pd.Series(dtype=str)) == "REVISAR").sum()) if not salary_rev.empty else 0},
        {"Indicador": "Auxilio / BigPass con diferencia", "Valor": int((const_rev.get("Estado", pd.Series(dtype=str)) == "REVISAR").sum()) if not const_rev.empty else 0},
        {"Indicador": "Seguridad social con diferencia", "Valor": int((ss_rev.get("Estado", pd.Series(dtype=str)) == "REVISAR").sum()) if not ss_rev.empty else 0},
        {"Indicador": "Lunes a viernes del mes", "Valor": int(calendar_df["Cuenta ZH base L-V"].sum())},
        {"Indicador": "Sábados del mes", "Valor": int(calendar_df["Es sábado"].sum())},
        {"Indicador": "Domingos del mes", "Valor": int(calendar_df["Es domingo"].sum())},
        {"Indicador": "Lunes festivos", "Valor": int(calendar_df["Es lunes festivo"].sum())},
        {"Indicador": "Log advertencias", "Valor": len([x for x in logs if x.get("Tipo") == "Advertencia"])},
    ])


def export_excel(sheets: Dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter", datetime_format="dd/mm/yyyy", date_format="dd/mm/yyyy") as writer:
        for name, df in sheets.items():
            safe_name = name[:31]
            if df is None:
                df = pd.DataFrame()
            df.to_excel(writer, index=False, sheet_name=safe_name)
            wb = writer.book
            ws = writer.sheets[safe_name]
            header_fmt = wb.add_format({"bold": True, "font_color": "white", "bg_color": "#00843D", "border": 1})
            money_fmt_x = wb.add_format({"num_format": "$#,##0", "border": 1})
            num_fmt = wb.add_format({"num_format": "#,##0.00", "border": 1})
            ok_fmt = wb.add_format({"bg_color": "#E2F0D9", "font_color": "#006100"})
            rev_fmt = wb.add_format({"bg_color": "#FCE4D6", "font_color": "#9C0006"})
            for col_num, value in enumerate(df.columns):
                ws.write(0, col_num, value, header_fmt)
                width = min(max(len(str(value)) + 2, 12), 42)
                sample = df[value].astype(str).head(60).map(len).max() if not df.empty else 0
                width = min(max(width, int(sample) + 2 if pd.notna(sample) else width), 48)
                ws.set_column(col_num, col_num, width)
                if any(k in norm_text(value) for k in ["valor", "salario", "base", "diferencia", "importe", "pension", "salud", "vacaciones", "ibc"]):
                    ws.set_column(col_num, col_num, width, money_fmt_x)
                elif any(k in norm_text(value) for k in ["dias", "cantidad", "jornada"]):
                    ws.set_column(col_num, col_num, width, num_fmt)
            if not df.empty:
                ws.autofilter(0, 0, len(df), max(len(df.columns) - 1, 0))
                ws.freeze_panes(1, 0)
                if "Estado" in df.columns:
                    col = df.columns.get_loc("Estado")
                    ws.conditional_format(1, col, len(df), col, {"type": "text", "criteria": "containing", "value": "OK", "format": ok_fmt})
                    ws.conditional_format(1, col, len(df), col, {"type": "text", "criteria": "containing", "value": "REVISAR", "format": rev_fmt})
    bio.seek(0)
    return bio.getvalue()

# =============================
# UI
# =============================
def inject_css():
    st.markdown(
        """
        <style>
        .stApp {background: linear-gradient(180deg, #F7FFF9 0%, #FFFFFF 38%, #FFF9F2 100%);}
        .hero {
            background: linear-gradient(135deg, #00843D 0%, #00A859 42%, #F28C28 100%);
            padding: 26px 28px;
            border-radius: 24px;
            color: white;
            box-shadow: 0 12px 32px rgba(0,0,0,.13);
            margin-bottom: 18px;
        }
        .hero h1 {margin: 0; font-size: 2.15rem;}
        .hero p {margin: 8px 0 0 0; font-size: 1.02rem; opacity: .96;}
        .pt60 {background:#FFF3CD; border-left:8px solid #F28C28; padding:16px 18px; border-radius:16px; color:#5C3B00; margin-bottom:18px;}
        .note {background:#EAF7EF; border-left:8px solid #00843D; padding:14px 16px; border-radius:14px; color:#0B4D2B; margin: 8px 0 16px 0;}
        .footer {text-align:center; color:#49624f; padding-top:24px; font-size:.88rem;}
        div[data-testid="stMetricValue"] { color:#00843D; }
        .stButton>button {border-radius: 14px; font-weight: 700;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def sheet_selector(label: str, file, kind: str) -> Optional[str]:
    sheets = list_sheets(file) if file else []
    if not sheets:
        return None
    default = 0
    preferences = {
        "preno": ["preno", "convertida"],
        "md": ["masterdata", "md"],
        "conceptos": ["hoja", "concept"],
    }.get(kind, [])
    for i, s in enumerate(sheets):
        if any(p in norm_text(s) for p in preferences):
            default = i
            break
    return st.selectbox(label, sheets, index=default)


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
    inject_css()
    st.markdown(f"""
    <div class="hero">
      <h1>{APP_ICON} {APP_TITLE}</h1>
      <p>Motor integral de revisión Part Time ZH/ZP · Vacaciones · Salario · Auxilio · BigPass · IBC y Seguridad Social</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="pt60"><b>⚠️ Recordatorio importante:</b><br>
    No olvides correr tiempos de los Part Time a través de la <b>PT60</b> antes de calcular.</div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="note"><b>Alcance:</b> la app evalúa únicamente empleados <b>ZH</b> y <b>ZP</b>, pero para ellos revisa todos los módulos: vacaciones, salario, auxilio, BigPass, bases 9262/9263, salud Z000 y pensión Z010.</div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Parámetros")
        today = date.today()
        year = int(st.number_input("Año de revisión", min_value=2023, max_value=2035, value=today.year, step=1))
        month_name = st.selectbox("Mes de revisión", list(MESES_ES.values()), index=today.month - 1)
        month = list(MESES_ES.values()).index(month_name) + 1
        st.divider()
        st.subheader("Constantes")
        aux_value = float(st.number_input("Auxilio de transporte mensual", min_value=0.0, value=200000.0, step=1000.0, format="%.0f"))
        bigpass_value = float(st.number_input("BigPass mensual", min_value=0.0, value=175000.0, step=1000.0, format="%.0f"))
        tolerance_money = float(st.number_input("Tolerancia diferencias ($)", min_value=0.0, value=100.0, step=50.0, format="%.0f"))
        tolerance_days = float(st.number_input("Tolerancia días", min_value=0.0, value=0.01, step=0.01, format="%.2f"))
        health_rate = float(st.number_input("% Salud empleado", min_value=0.0, max_value=0.2, value=0.04, step=0.005, format="%.3f"))
        pension_rate = float(st.number_input("% Pensión empleado", min_value=0.0, max_value=0.2, value=0.04, step=0.005, format="%.3f"))
        st.divider()
        st.subheader("Promedio vacaciones")
        absence_families_base = st.multiselect(
            "Ausentismos que hacen base de promedio",
            options=["INCAPACIDAD", "REMUNERADA", "VACACIONES", "OTRA", "ANJ", "LNR", "SUSPENSION"],
            default=["INCAPACIDAD", "REMUNERADA"],
            help="Se valorizan con salario histórico vigente y días reales según ZH/ZP."
        )
        with st.expander("🔧 Códigos"):
            aux_codes = [x.strip().upper() for x in st.text_input("Auxilio", ",".join(AUX_CODES_DEFAULT)).split(",") if x.strip()]
            bp_codes = [x.strip().upper() for x in st.text_input("BigPass ingreso", ",".join(BIGPASS_CODES_DEFAULT)).split(",") if x.strip()]
            bp_ded_codes = [x.strip().upper() for x in st.text_input("BigPass descuento", ",".join(BIGPASS_DED_CODES_DEFAULT)).split(",") if x.strip()]
            excluded_ss = [x.strip().upper() for x in st.text_input("Excluir de IBC app", ",".join(NON_SALARY_DEFAULT)).split(",") if x.strip()]

    st.subheader("1. Carga de archivos")
    c1, c2, c3 = st.columns(3)
    with c1:
        preno_file = st.file_uploader("📄 Prenómina convertida Excel", type=["xlsx", "xls", "xlsm"], key="preno_xlsx")
        md_file = st.file_uploader("👥 Master Data Global / MD PT", type=["xlsx", "xls", "xlsm", "txt"], key="md")
        base662_file = st.file_uploader("🧾 /662 mes anterior (opcional)", type=["xlsx", "xls", "xlsm", "txt"], key="base662")
    with c2:
        acum_file = st.file_uploader("📚 Acumulados últimos 12 meses", type=["xlsx", "xls", "xlsm", "txt"], key="acum")
        aus_file = st.file_uploader("🗓️ AUSNOM acumulado último año", type=["xlsx", "xls", "xlsm", "txt"], key="aus")
    with c3:
        conceptos_file = st.file_uploader("📌 Matriz conceptos base variable", type=["xlsx", "xls", "xlsm", "txt"], key="conceptos")
        hist_file = st.file_uploader("💰 Histórico salarios PT", type=["xlsx", "xls", "xlsm", "txt"], key="hist")
        st.caption("La prenómina debe ser la convertida a Excel. El TXT de recibos SAP sirve como fuente previa, no como prenómina principal de este motor.")

    required = [preno_file, md_file, acum_file, aus_file, conceptos_file, hist_file]
    if not all(required):
        st.info("Carga los archivos obligatorios para ejecutar el motor integral.")
        cal = build_calendar(year, month)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("L-V", int(cal["Cuenta ZH base L-V"].sum()))
        k2.metric("Sábados", int(cal["Es sábado"].sum()))
        k3.metric("Domingos", int(cal["Es domingo"].sum()))
        k4.metric("Lunes festivos", int(cal["Es lunes festivo"].sum()))
        with st.expander("Ver calendario Colombia"):
            st.dataframe(cal, use_container_width=True)
        st.markdown('<div class="footer">Creado por Andrés Huérfano Dávila – Nómina JMC</div>', unsafe_allow_html=True)
        return

    st.subheader("2. Hojas a usar")
    s1, s2, s3 = st.columns(3)
    with s1:
        preno_sheet = sheet_selector("Hoja prenómina", preno_file, "preno")
        md_sheet = sheet_selector("Hoja MD", md_file, "md")
    with s2:
        conceptos_sheet = sheet_selector("Hoja conceptos", conceptos_file, "conceptos")
    with s3:
        # Los TXT no requieren hoja.
        pass

    run = st.button("🚀 Ejecutar motor integral Part Time", type="primary", use_container_width=True)
    if not run:
        st.markdown('<div class="footer">Creado por Andrés Huérfano Dávila – Nómina JMC</div>', unsafe_allow_html=True)
        return

    logs: List[Dict[str, str]] = []
    try:
        with st.spinner("Leyendo archivos y ejecutando validaciones..."):
            raw_preno = read_any_table(preno_file, "preno", preno_sheet)
            raw_md = read_any_table(md_file, "md", md_sheet)
            raw_acum = read_any_table(acum_file, "acum")
            raw_aus = read_any_table(aus_file, "ausnom")
            raw_conceptos = read_any_table(conceptos_file, "conceptos", conceptos_sheet)
            raw_hist = read_any_table(hist_file, "hist")
            raw_662 = read_any_table(base662_file, "662") if base662_file else pd.DataFrame()

            md = prepare_md(raw_md)
            preno, _ = prepare_preno(raw_preno)
            # Filtra prenómina solo empleados PT del MD.
            preno = preno[preno["SAP"].isin(set(md["SAP"]))].copy()
            acum = prepare_acum(raw_acum)
            acum = acum[acum["SAP"].isin(set(md["SAP"]))].copy()
            aus = prepare_ausnom(raw_aus, md)
            aus = aus[aus["SAP"].isin(set(md["SAP"]))].copy()
            hist = prepare_hist(raw_hist)
            hist = hist[hist["SAP"].isin(set(md["SAP"]))].copy()
            variable_codes = prepare_variable_codes(raw_conceptos)
            base662 = prepare_662(raw_662) if base662_file else pd.DataFrame(columns=["SAP", "Valor /662", "Cantidad"])
            calendar_df = build_calendar(year, month)

            aus_month = build_absence_month(aus, year, month, tolerance_days)
            base_vac, dev_detail, abs_detail = build_vacation_module(
                acum=acum, aus=aus, md=md, hist=hist, preno=preno,
                variable_codes=variable_codes, year=year, month=month,
                tolerance_money=tolerance_money, absence_families_base=absence_families_base,
            )
            salary_rev = build_salary_review(preno, md, hist, year, month, tolerance_money)
            const_rev = build_constants_review(preno, md, aux_value, bigpass_value, tolerance_money, aux_codes, bp_codes, bp_ded_codes)
            ss_rev = build_ibc_ss_review(preno, md, aus_month, base662, variable_codes, excluded_ss, health_rate, pension_rate, tolerance_money)

            logs.append({"Tipo": "OK", "Detalle": "Proceso ejecutado correctamente"})
            logs.append({"Tipo": "Parámetro", "Detalle": f"Período: {MESES_ES[month]} {year}"})
            logs.append({"Tipo": "Parámetro", "Detalle": f"Días base constante: {DIAS_BASE}"})
            logs.append({"Tipo": "Parámetro", "Detalle": f"Conceptos variables base: {', '.join(variable_codes)}"})
            if base662_file is None:
                logs.append({"Tipo": "Advertencia", "Detalle": "No se cargó /662. La hoja IBC_SS se calcula sin referencia de base mes anterior para incapacidades."})
            if preno.empty:
                logs.append({"Tipo": "Advertencia", "Detalle": "La prenómina quedó vacía después de filtrar empleados PT del MD."})

            resumen = build_summary(md, preno, acum, aus, aus_month, base_vac, salary_rev, const_rev, ss_rev, calendar_df, logs)
            log_df = pd.DataFrame(logs)

        st.success("Motor ejecutado correctamente.")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("PT evaluados", len(md))
        m2.metric("Vacaciones", len(base_vac))
        m3.metric("Dif. salario", int((salary_rev.get("Estado", pd.Series(dtype=str)) == "REVISAR").sum()) if not salary_rev.empty else 0)
        m4.metric("Dif. aux/BP", int((const_rev.get("Estado", pd.Series(dtype=str)) == "REVISAR").sum()) if not const_rev.empty else 0)
        m5.metric("Dif. SS", int((ss_rev.get("Estado", pd.Series(dtype=str)) == "REVISAR").sum()) if not ss_rev.empty else 0)

        tabs = st.tabs(["Resumen", "Vacaciones", "Salario", "Aux/BigPass", "IBC/SS", "Ausentismos", "Calendario", "Log"])
        with tabs[0]: st.dataframe(resumen, use_container_width=True)
        with tabs[1]: st.dataframe(base_vac, use_container_width=True)
        with tabs[2]: st.dataframe(salary_rev, use_container_width=True)
        with tabs[3]: st.dataframe(const_rev, use_container_width=True)
        with tabs[4]: st.dataframe(ss_rev, use_container_width=True)
        with tabs[5]: st.dataframe(aus_month, use_container_width=True)
        with tabs[6]: st.dataframe(calendar_df, use_container_width=True)
        with tabs[7]: st.dataframe(log_df, use_container_width=True)

        sheets = {
            "Resumen": resumen,
            "Revision_Vacaciones": base_vac,
            "Detalle_Devengos_Base": dev_detail,
            "Detalle_Ausent_Base": abs_detail,
            "Revision_Salario": salary_rev,
            "Revision_Aux_BigPass": const_rev,
            "Revision_IBC_SS": ss_rev,
            "Ausentismos_Mes": aus_month,
            "Calendario_CO": calendar_df,
            "MD_PT": md,
            "Prenomina_Normalizada": preno,
            "Log_Proceso": log_df,
        }
        excel = export_excel(sheets)
        fname = f"revision_integral_part_time_{year}_{month:02d}.xlsx"
        st.download_button("⬇️ Descargar Excel de revisión integral", data=excel, file_name=fname, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

    except Exception as e:
        st.error("No fue posible completar el proceso.")
        st.exception(e)
        st.info("Revisa que estés usando la prenómina convertida en Excel y que AUSNOM sea acumulado del último año.")

    st.markdown('<div class="footer">Creado por Andrés Huérfano Dávila – Nómina JMC</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
