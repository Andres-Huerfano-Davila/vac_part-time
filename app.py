# -*- coding: utf-8 -*-
"""
Auditor Nómina Part Time - JMC
Creado por Andrés Huérfano Dávila – Nómina JMC

Versión v5
- Módulo 1: base promedio y pago esperado de vacaciones Part Time.
- Módulo 2: comparación contra prenómina SAP.
- AUSNOM acumulado: se usa sobre la ventana de 365 días, no solo el mes.
- Histórico de salarios PT: respaldo oficial cuando no hay salario pagado en acumulados.
- Calendario interno Colombia.
- ZH/ZP con reglas diferenciales.
"""
from __future__ import annotations

import calendar
import io
import re
import unicodedata
import zipfile
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
# Parámetros generales
# =============================
APP_TITLE = "Auditor Nómina Part Time"
APP_ICON = "🦜"
DIAS_BASE = 30
HORAS_DIA_ZH = 4

MESES_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}
DIAS_ES = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

DEFAULT_AUX_CODES = ["Y200"]
DEFAULT_BIGPASS_CODES = ["Y598"]
DEFAULT_BIGPASS_DED_CODES = ["Z590"]
DEFAULT_VAC_CODES = ["Y400"]
SALARY_CODES_ZH = ["Y090"]
SALARY_CODES_ZP = ["Y011"]
BONUS_ANT_CODES = ["Y617"]
SALARY_ALL_CODES = set(SALARY_CODES_ZH + SALARY_CODES_ZP + BONUS_ANT_CODES)
DEFAULT_ABS_BASE_FAMILIES = ["INCAPACIDAD", "REMUNERADA", "VACACIONES"]

# =============================
# Utilidades de limpieza
# =============================
def norm_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    txt = str(value).strip().lower()
    txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii")
    txt = re.sub(r"\s+", " ", txt)
    return txt


def clean_employee_id(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.strip()
    )


def safe_num(series) -> pd.Series:
    """Convierte números SAP/Excel con formato colombiano.
    Ejemplos: 1.463.000 -> 1463000; 9,00 -> 9.0; 1.234,56 -> 1234.56.
    """
    def conv(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return 0.0
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        s = str(x).strip().replace("\u00a0", "").replace(" ", "")
        if s == "" or s.lower() in ["nan", "none"]:
            return 0.0
        neg = s.startswith("-")
        s2 = s[1:] if neg else s
        if "," in s2 and "." in s2:
            # 1.234.567,89
            s2 = s2.replace(".", "").replace(",", ".")
        elif "," in s2:
            # 9,00 / 1234,5
            s2 = s2.replace(".", "").replace(",", ".")
        elif "." in s2:
            # Si son puntos de miles: 438.900, 1.463.000
            parts = s2.split(".")
            if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3):
                s2 = "".join(parts)
            # Si no, se conserva como decimal anglosajón.
        s2 = re.sub(r"[^0-9.]", "", s2)
        try:
            val = float(s2) if s2 else 0.0
            return -val if neg else val
        except Exception:
            return 0.0
    return pd.Series(series).apply(conv).astype(float)


def parse_date_value(x) -> Optional[date]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    txt = str(x).strip()
    if txt in ["", "nan", "NaT"]:
        return None
    if txt in ["31.12.9999", "31/12/9999", "9999-12-31"]:
        return date(2099, 12, 31)
    for fmt in ["%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"]:
        try:
            return datetime.strptime(txt[:10], fmt).date()
        except Exception:
            pass
    val = pd.to_datetime(txt, errors="coerce", dayfirst=True)
    if pd.isna(val):
        return None
    return val.date()


def safe_date(series) -> pd.Series:
    return pd.Series(series).apply(parse_date_value)


def unique_columns(cols: Iterable[str]) -> List[str]:
    seen = {}
    out = []
    for c in cols:
        base = str(c).strip() or "Col"
        if base in seen:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
        else:
            seen[base] = 1
            out.append(base)
    return out


def find_col(df: pd.DataFrame, candidates: List[str], contains: bool = False) -> Optional[str]:
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    norm_map = {norm_text(c): c for c in cols}
    for cand in candidates:
        nc = norm_text(cand)
        if nc in norm_map:
            return norm_map[nc]
    for cand in candidates:
        nc = norm_text(cand)
        for col in cols:
            ncol = norm_text(col)
            if (contains and nc in ncol) or (not contains and (nc == ncol or nc in ncol)):
                return col
    return None


def find_code_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if "cc" in norm_text(c) or "cod" in norm_text(c) or "concept" in norm_text(c)]
    best = None
    best_score = -1
    for c in candidates:
        vals = df[c].astype(str).str.strip().str.upper()
        score = vals.str.match(r"^(Y|Z|/|9)[A-Z0-9/]{2,5}$", na=False).sum()
        if score > best_score:
            best_score = score
            best = c
    return best

# =============================
# Lectura de archivos
# =============================
def get_bytes(uploaded_file) -> bytes:
    if uploaded_file is None:
        return b""
    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()
    with open(uploaded_file, "rb") as f:
        return f.read()


def file_name(uploaded_file) -> str:
    return getattr(uploaded_file, "name", str(uploaded_file))


def is_excel_file(uploaded_file) -> bool:
    n = file_name(uploaded_file).lower()
    return n.endswith((".xlsx", ".xlsm", ".xls", ".xlsb"))


def list_sheets(uploaded_file) -> List[str]:
    if uploaded_file is None or not is_excel_file(uploaded_file):
        return ["Archivo TXT/CSV"]
    data = get_bytes(uploaded_file)
    return pd.ExcelFile(io.BytesIO(data)).sheet_names


def read_sap_pipe_text(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.count("|") >= 3:
            low = norm_text(line)
            if "pers" in low or "cc-n" in low or "valido" in low or "fecha pago" in low:
                header_idx = i
                break
    if header_idx is None:
        raise ValueError("No se encontró encabezado tipo SAP con separadores '|'.")

    headers = [x.strip() for x in lines[header_idx].strip().strip("|").split("|")]
    cols = unique_columns(headers)
    rows = []
    for line in lines[header_idx + 1:]:
        if not line.strip().startswith("|"):
            continue
        parts = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(parts) != len(cols):
            continue
        if all((p == "" or set(p) <= set("- ")) for p in parts):
            continue
        rows.append(parts)
    if not rows:
        raise ValueError("No se encontraron filas útiles en el archivo SAP.")
    return pd.DataFrame(rows, columns=cols)


def read_text_table(uploaded_file) -> pd.DataFrame:
    data = get_bytes(uploaded_file)
    text = None
    for enc in ["utf-8-sig", "latin-1", "cp1252"]:
        try:
            text = data.decode(enc)
            break
        except Exception:
            pass
    if text is None:
        raise ValueError("No fue posible leer el archivo de texto.")
    if "|" in text:
        try:
            return read_sap_pipe_text(text)
        except Exception:
            pass
    # Fallback delimitado
    for sep in ["\t", ";", ",", "|"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, engine="python")
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    raise ValueError("No se pudo identificar el delimitador del archivo de texto.")


def read_table(uploaded_file, sheet_name: Optional[str] = None) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    if is_excel_file(uploaded_file):
        data = get_bytes(uploaded_file)
        sheet = 0 if sheet_name in [None, "Archivo TXT/CSV"] else sheet_name
        return pd.read_excel(io.BytesIO(data), sheet_name=sheet, dtype=object)
    return read_text_table(uploaded_file)

# =============================
# Calendario Colombia
# =============================
def fallback_colombia_holidays(year: int) -> Dict[date, str]:
    # Respaldo mínimo; si la librería holidays está instalada, se usa la lista oficial calculada.
    fixed = {
        date(year, 1, 1): "Año Nuevo",
        date(year, 5, 1): "Día del Trabajo",
        date(year, 7, 20): "Independencia de Colombia",
        date(year, 8, 7): "Batalla de Boyacá",
        date(year, 12, 8): "Inmaculada Concepción",
        date(year, 12, 25): "Navidad",
    }
    return fixed


def get_colombia_holidays(year: int) -> Dict[date, str]:
    if holidays is not None:
        try:
            return {d: name for d, name in holidays.Colombia(years=[year]).items()}
        except Exception:
            pass
    return fallback_colombia_holidays(year)


def build_calendar(year: int, month: int) -> pd.DataFrame:
    hols = get_colombia_holidays(year)
    rows = []
    for day in range(1, calendar.monthrange(year, month)[1] + 1):
        d = date(year, month, day)
        wd = d.weekday()
        is_holiday = d in hols
        rows.append({
            "Fecha": d,
            "Día semana": DIAS_ES[wd],
            "Es festivo Colombia": is_holiday,
            "Festivo": hols.get(d, ""),
            "Es sábado": wd == 5,
            "Es domingo": wd == 6,
            "Es lunes festivo": wd == 0 and is_holiday,
            "Cuenta ZH base L-V": wd <= 4,
            "Cuenta ZP FDS": wd in (5, 6) or (wd == 0 and is_holiday),
        })
    return pd.DataFrame(rows)


def dates_between(start: date, end: date) -> List[date]:
    if start is None or end is None or end < start:
        return []
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def jornada_mensual_vigente(d: date) -> int:
    if d is None:
        return 220
    if d < date(2025, 7, 15):
        return 230
    if d < date(2026, 7, 15):
        return 220
    return 210

# =============================
# Reglas Part Time
# =============================
def area_group(area_text: str) -> str:
    t = norm_text(area_text)
    if t == "zh" or "tiempor parcial hora" in t or "tiempo parcial hora" in t or "parcial hora" in t:
        return "ZH"
    if t == "zp" or "tiempo parcial dia" in t or "parcial dia" in t or "fds" in t:
        return "ZP"
    return "NO_IDENTIFICADA"


def absence_family(abs_text: str) -> str:
    t = norm_text(abs_text)
    if "vacacion" in t or "vacaciones" in t:
        return "VACACIONES"
    if "ausencia no just" in t or " no justificada" in t or t.startswith("anj") or "aus reg sin soporte" in t or "sin soporte" in t:
        return "ANJ"
    if "licencia no remunerada" in t or "lic no remuner" in t or "unpaid" in t:
        return "LNR"
    if "suspension" in t:
        return "SUSPENSION"
    if "incap" in t or "inca" in t or "prorroga" in t or "enfermedad" in t or "accid" in t:
        return "INCAPACIDAD"
    if "dia de la familia" in t or "calamidad" in t or "luto" in t or "licencia remunerada" in t:
        return "REMUNERADA"
    return "OTRA"


def date_counts_for_rule(abs_family: str, area: str, start: date, end: date, window_start: date, window_end: date) -> Tuple[int, List[date], str]:
    if start is None or end is None:
        return 0, [], "Fecha inválida"
    adj_start = max(start, window_start)
    adj_end = min(end, window_end)
    if adj_end < adj_start:
        return 0, [], "Fuera de ventana"
    years = list(range(adj_start.year, adj_end.year + 1))
    hols = {}
    for y in years:
        hols.update(get_colombia_holidays(y))
    selected = []
    for d in dates_between(adj_start, adj_end):
        wd = d.weekday()
        is_holiday = d in hols
        include = False
        if area == "ZH":
            if abs_family in ["VACACIONES", "LNR", "SUSPENSION"]:
                include = wd <= 4 and not is_holiday
            elif abs_family in ["ANJ", "INCAPACIDAD", "REMUNERADA", "OTRA"]:
                include = wd <= 4
        elif area == "ZP":
            if abs_family == "VACACIONES":
                include = wd <= 5 and not is_holiday
            elif abs_family in ["LNR", "ANJ", "INCAPACIDAD", "REMUNERADA", "OTRA"]:
                include = wd in (5, 6) or (wd == 0 and is_holiday)
            elif abs_family == "SUSPENSION":
                include = wd == 5
        if include:
            selected.append(d)
    notes = []
    if start < window_start or end > window_end:
        notes.append("Ausencia cruza la ventana; se calculó solo el tramo aplicable")
    return len(selected), selected, "; ".join(notes)

# =============================
# Normalización de bases
# =============================
def prepare_md(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["SAP", "Nombre MD", "Área nómina MD", "Grupo área", "CECO", "Cargo"])
    sap_col = find_col(df, ["Nº pers.", "N° pers.", "Número de personal", "Numero de personal", "SAP", "Nº personal"])
    name_col = find_col(df, ["Nombre", "Número de personal", "Nom.empl./cand.", "Nombre completo", "Nombre del empleado"], contains=True)
    area_col = find_col(df, ["Área de nómina", "Área nóm.", "Area de nomina", "Texto área nómina", "Área cálculo nómina"])
    ceco_col = find_col(df, ["Ce.coste", "CECO", "Centro de coste", "Centro de costo"])
    cargo_col = find_col(df, ["Cargo", "Posición", "Denom.función", "Función"])
    if not sap_col:
        raise ValueError("No se encontró columna SAP/Nº pers. en MD.")
    out = pd.DataFrame()
    out["SAP"] = clean_employee_id(df[sap_col])
    out["Nombre MD"] = df[name_col].astype(str) if name_col else ""
    out["Área nómina MD"] = df[area_col].astype(str) if area_col else ""
    out["Grupo área"] = out["Área nómina MD"].apply(area_group)
    out["CECO"] = df[ceco_col].astype(str) if ceco_col else ""
    out["Cargo"] = df[cargo_col].astype(str) if cargo_col else ""
    out = out[out["SAP"].str.len() > 0].drop_duplicates("SAP", keep="last")
    return out


def prepare_preno(df: pd.DataFrame) -> pd.DataFrame:
    code_col = find_col(df, ["CÓDIGO", "CODIGO", "Código", "CC-nómina", "CC-n."]) or find_code_col(df)
    concept_col = find_col(df, ["CONCEPTO", "Texto expl.CC-nómina", "Concepto", "CC-nómina_2"])
    qty_col = find_col(df, ["CANTIDAD", "CANT", "Cantidad", "Ctd."])
    val_col = find_col(df, ["VALOR", "Importe", "Valor"])
    sap_col = find_col(df, ["SAP", "Nº pers.", "N° pers.", "Número de personal", "Numero de personal"])
    name_col = find_col(df, ["NOMBRE", "Nombre", "Nom.empl./cand."])
    if not all([code_col, val_col, sap_col]):
        raise ValueError("No se pudo identificar Código, Valor y SAP en la prenómina.")
    out = pd.DataFrame()
    out["SAP"] = clean_employee_id(df[sap_col])
    out["Código"] = df[code_col].astype(str).str.strip().str.upper()
    out["Concepto"] = df[concept_col].astype(str) if concept_col else ""
    out["Cantidad SAP"] = safe_num(df[qty_col]) if qty_col else 0.0
    out["Valor SAP"] = safe_num(df[val_col])
    out["Nombre Preno"] = df[name_col].astype(str) if name_col else ""
    return out[out["SAP"].str.len() > 0].copy()


def prepare_acum(df: pd.DataFrame) -> pd.DataFrame:
    sap_col = find_col(df, ["Nº pers.", "N° pers.", "Número de personal", "Numero de personal", "SAP"])
    date_col = find_col(df, ["Fecha pago", "Fecha de pago", "Periodo", "Período"])
    code_col = find_col(df, ["CC-n.", "CC-nómina", "CC nomina", "Código", "CODIGO"]) or find_code_col(df)
    concept_col = find_col(df, ["Texto expl.CC-nómina", "Concepto", "CC-nómina_2"])
    qty_col = find_col(df, ["Cantidad", "CANTIDAD", "Ctd.", "CANT"])
    amount_col = find_col(df, ["Importe", "VALOR", "Valor"])
    if not all([sap_col, date_col, code_col, amount_col]):
        raise ValueError("No se pudo identificar SAP, Fecha pago, Código e Importe en acumulados.")
    out = pd.DataFrame()
    out["SAP"] = clean_employee_id(df[sap_col])
    out["Fecha pago"] = safe_date(df[date_col])
    out["Código"] = df[code_col].astype(str).str.strip().str.upper()
    out["Concepto"] = df[concept_col].astype(str) if concept_col else ""
    out["Cantidad"] = safe_num(df[qty_col]) if qty_col else 0.0
    out["Importe"] = safe_num(df[amount_col])
    out = out[(out["SAP"].str.len() > 0) & out["Fecha pago"].notna()].copy()
    return out


def prepare_ausnom(df: pd.DataFrame, md: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    sap_col = find_col(df, ["Nº pers.", "N° pers.", "Nº personal", "Número de personal", "Numero de personal", "SAP"])
    name_col = find_col(df, ["Nom.empl./cand.", "Nombre", "Nombre completo", "Nombre del empleado"])
    abs_col = find_col(df, ["Txt.cl.pres./ab.", "Ausentismo", "Clase absent", "cl.pres", "Texto"])
    start_col = find_col(df, ["Válido de", "Valido de", "Inicio de validez", "Fecha inicio", "Desde", "Fecha de"])
    end_col = find_col(df, ["Válido a", "Valido a", "Fin de validez", "Fecha fin", "Hasta", "Fecha a"])
    days_col = find_col(df, ["D.ab./pr.", "Días presenc./abs.", "Dias presenc", "Días", "Dias"])
    nat_col = find_col(df, ["Día nat.", "Dias nat", "Días naturales"])
    area_col = find_col(df, ["Área nóm.", "Área de nómina", "Area nomina", "Área cálculo nómina"])
    if not all([sap_col, abs_col, start_col, end_col]):
        raise ValueError("No se pudo identificar SAP, ausentismo, fecha inicio y fecha fin en AUSNOM.")
    out = pd.DataFrame()
    out["SAP"] = clean_employee_id(df[sap_col])
    out["Nombre Ausnom"] = df[name_col].astype(str) if name_col else ""
    out["Ausentismo SAP"] = df[abs_col].astype(str)
    out["Familia ausencia"] = out["Ausentismo SAP"].apply(absence_family)
    out["Fecha inicio"] = safe_date(df[start_col])
    out["Fecha fin"] = safe_date(df[end_col])
    out["Días SAP"] = safe_num(df[days_col]) if days_col else 0.0
    out["Días naturales SAP"] = safe_num(df[nat_col]) if nat_col else 0.0
    out["Área nómina AUSNOM"] = df[area_col].astype(str) if area_col else ""
    out["Grupo área"] = out["Área nómina AUSNOM"].apply(area_group)
    if md is not None and not md.empty:
        out = out.merge(md[["SAP", "Nombre MD", "Área nómina MD", "Grupo área"]].rename(columns={"Grupo área":"Grupo área MD"}), on="SAP", how="left")
        out["Grupo área"] = np.where(out["Grupo área"].eq("NO_IDENTIFICADA"), out["Grupo área MD"].fillna("NO_IDENTIFICADA"), out["Grupo área"])
    return out[out["SAP"].str.len() > 0].copy()


def prepare_hist_salarios(df: pd.DataFrame) -> pd.DataFrame:
    sap_col = find_col(df, ["Nº pers.", "N° pers.", "Número de personal", "Numero de personal", "SAP"])
    name_col = find_col(df, ["Número de personal", "Nombre", "Nom.empl./cand."])
    area_col = find_col(df, ["Área de nómina", "Área nóm.", "Area de nomina"])
    code_col = find_code_col(df) or find_col(df, ["CC-nómina", "CC-n."])
    amount_col = find_col(df, ["Importe", "Salario", "Valor"])
    # Histórico tiene dos columnas Desde; la segunda suele quedar como Desde_2.
    desde_cols = [c for c in df.columns if norm_text(c).startswith("desde")]
    start_col = desde_cols[-1] if desde_cols else find_col(df, ["Desde"])
    end_col = find_col(df, ["Hasta"])
    if not all([sap_col, area_col, code_col, amount_col, start_col, end_col]):
        raise ValueError("No se pudo identificar SAP, área, código, importe y vigencias en histórico de salarios.")
    out = pd.DataFrame()
    out["SAP"] = clean_employee_id(df[sap_col])
    out["Nombre Histórico"] = df[name_col].astype(str) if name_col else ""
    out["Área nómina Histórico"] = df[area_col].astype(str)
    out["Grupo área"] = out["Área nómina Histórico"].apply(area_group)
    out["Código"] = df[code_col].astype(str).str.strip().str.upper()
    out["Importe vigente"] = safe_num(df[amount_col])
    out["Desde vigencia"] = safe_date(df[start_col])
    out["Hasta vigencia"] = safe_date(df[end_col])
    out = out[(out["SAP"].str.len() > 0) & out["Desde vigencia"].notna()].copy()
    return out


def prepare_conceptos_variables(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return []
    code_col = find_code_col(df) or find_col(df, ["CC-nómina", "CC-n.", "Código", "CODIGO", "Concepto"])
    if not code_col:
        return []
    codes = df[code_col].astype(str).str.strip().str.upper()
    codes = [c for c in codes if re.match(r"^(Y|Z|/|9)[A-Z0-9/]{2,5}$", c)]
    return sorted(set([c for c in codes if c not in SALARY_ALL_CODES]))

# =============================
# Salario vigente y valorización
# =============================
def get_salary_from_hist(hist: pd.DataFrame, sap: str, d: date, area: str) -> Dict[str, object]:
    empty = {"salario_base": 0.0, "bono_ant": 0.0, "salario_total": 0.0, "valor_dia_pt": 0.0, "fuente": "No encontrado", "detalle": ""}
    if hist is None or hist.empty or d is None:
        return empty
    tmp = hist[(hist["SAP"] == str(sap)) & (hist["Desde vigencia"] <= d) & (hist["Hasta vigencia"] >= d)].copy()
    if tmp.empty:
        return empty
    if area == "ZH":
        salary_codes = SALARY_CODES_ZH
    elif area == "ZP":
        salary_codes = SALARY_CODES_ZP
    else:
        salary_codes = SALARY_CODES_ZH + SALARY_CODES_ZP
    sal_base = tmp[tmp["Código"].isin(salary_codes)]["Importe vigente"].sum()
    bono = tmp[tmp["Código"].isin(BONUS_ANT_CODES)]["Importe vigente"].sum()
    total = sal_base + bono
    if total <= 0:
        return empty
    if area == "ZH":
        jornada = jornada_mensual_vigente(d)
        valor_dia = total / jornada * HORAS_DIA_ZH
        detalle = f"Histórico salario {d.strftime('%d/%m/%Y')} · jornada {jornada}"
    else:
        valor_dia = total / DIAS_BASE
        detalle = f"Histórico salario {d.strftime('%d/%m/%Y')} · base 30"
    return {"salario_base": sal_base, "bono_ant": bono, "salario_total": total, "valor_dia_pt": valor_dia, "fuente": "Histórico salarios PT", "detalle": detalle}


def get_salary_from_acum_month(acum: pd.DataFrame, sap: str, d: date, area: str) -> Dict[str, object]:
    empty = {"salario_base": 0.0, "bono_ant": 0.0, "salario_total": 0.0, "valor_dia_pt": 0.0, "fuente": "No encontrado", "detalle": ""}
    if acum is None or acum.empty or d is None:
        return empty
    tmp = acum[(acum["SAP"] == str(sap)) & (pd.Series(acum["Fecha pago"]).apply(lambda x: x.year == d.year and x.month == d.month))].copy()
    if tmp.empty:
        return empty
    if area == "ZH":
        sal_rows = tmp[tmp["Código"].isin(SALARY_CODES_ZH) & (tmp["Cantidad"] > 0) & (tmp["Importe"] > 0)]
        if sal_rows.empty:
            return empty
        # Si hay más de una línea, usamos valor hora ponderado.
        total_imp = sal_rows["Importe"].sum()
        total_qty = sal_rows["Cantidad"].sum()
        if total_qty <= 0:
            return empty
        valor_hora = total_imp / total_qty
        jornada = jornada_mensual_vigente(d)
        sal_base = valor_hora * jornada
        bono = tmp[tmp["Código"].isin(BONUS_ANT_CODES)]["Importe"].sum()
        total = sal_base + bono
        valor_dia = total / jornada * HORAS_DIA_ZH
        return {"salario_base": sal_base, "bono_ant": bono, "salario_total": total, "valor_dia_pt": valor_dia, "fuente": "Acumulados", "detalle": f"Y090 acumulados · jornada {jornada}"}
    if area == "ZP":
        sal_rows = tmp[tmp["Código"].isin(SALARY_CODES_ZP) & (tmp["Cantidad"] > 0) & (tmp["Importe"] > 0)]
        if sal_rows.empty:
            return empty
        total_imp = sal_rows["Importe"].sum()
        total_qty = sal_rows["Cantidad"].sum()
        if total_qty <= 0:
            return empty
        valor_dia = total_imp / total_qty
        sal_base = valor_dia * DIAS_BASE
        bono = tmp[tmp["Código"].isin(BONUS_ANT_CODES)]["Importe"].sum()
        total = sal_base + bono
        return {"salario_base": sal_base, "bono_ant": bono, "salario_total": total, "valor_dia_pt": total / DIAS_BASE, "fuente": "Acumulados", "detalle": "Y011 acumulados · base 30"}
    return empty


def get_salary_for_date(acum: pd.DataFrame, hist: pd.DataFrame, sap: str, d: date, area: str) -> Dict[str, object]:
    # 1. Acumulados del mes si existen.
    r = get_salary_from_acum_month(acum, sap, d, area)
    if r["salario_total"] > 0:
        return r
    # 2. Histórico como respaldo oficial.
    r = get_salary_from_hist(hist, sap, d, area)
    if r["salario_total"] > 0:
        return r
    # 3. Último salario válido anterior en acumulados.
    if acum is not None and not acum.empty:
        prev = acum[(acum["SAP"] == str(sap)) & (acum["Fecha pago"] <= d)].copy()
        if area == "ZH":
            prev = prev[prev["Código"].isin(SALARY_CODES_ZH) & (prev["Cantidad"] > 0) & (prev["Importe"] > 0)]
        elif area == "ZP":
            prev = prev[prev["Código"].isin(SALARY_CODES_ZP) & (prev["Cantidad"] > 0) & (prev["Importe"] > 0)]
        if not prev.empty:
            last_month = max(prev["Fecha pago"])
            r = get_salary_from_acum_month(acum, sap, last_month, area)
            if r["salario_total"] > 0:
                r["fuente"] = "Último acumulado anterior"
                r["detalle"] = f"No hubo salario en {d.strftime('%m/%Y')}; se usó {last_month.strftime('%m/%Y')}"
                return r
    return {"salario_base": 0.0, "bono_ant": 0.0, "salario_total": 0.0, "valor_dia_pt": 0.0, "fuente": "No encontrado", "detalle": "No se encontró salario en acumulados ni histórico"}

# =============================
# Módulo de vacaciones promedio
# =============================
def overlap(start: date, end: date, ws: date, we: date) -> bool:
    if start is None or end is None:
        return False
    return max(start, ws) <= min(end, we)


def build_vacation_base(
    acum: pd.DataFrame,
    aus_acum: pd.DataFrame,
    hist: pd.DataFrame,
    md: pd.DataFrame,
    variable_codes: List[str],
    year: int,
    month: int,
    abs_families_base: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    period_start = date(year, month, 1)
    period_end = date(year, month, calendar.monthrange(year, month)[1])

    aus = aus_acum.copy()
    if md is not None and not md.empty:
        # Reforzar área con MD cuando AUSNOM no la traiga.
        aus = aus.merge(md[["SAP", "Grupo área", "Nombre MD"]].rename(columns={"Grupo área": "Grupo área MD"}), on="SAP", how="left")
        aus["Grupo área"] = np.where(aus["Grupo área"].eq("NO_IDENTIFICADA"), aus["Grupo área MD"].fillna("NO_IDENTIFICADA"), aus["Grupo área"])
        aus["Nombre base"] = aus.get("Nombre MD", "").fillna(aus.get("Nombre Ausnom", ""))
    else:
        aus["Nombre base"] = aus.get("Nombre Ausnom", "")

    vacation_rows = aus[(aus["Familia ausencia"] == "VACACIONES") & aus.apply(lambda r: overlap(r["Fecha inicio"], r["Fecha fin"], period_start, period_end), axis=1)].copy()

    base_rows = []
    devengo_rows = []
    ausencia_rows = []
    salario_trace_rows = []

    var_codes = set([c.upper().strip() for c in variable_codes]) - SALARY_ALL_CODES
    abs_families = set(abs_families_base)

    for idx, vac in vacation_rows.iterrows():
        sap = vac["SAP"]
        area = vac["Grupo área"]
        vac_start = vac["Fecha inicio"]
        vac_end = vac["Fecha fin"]
        nombre = vac.get("Nombre base", "") or vac.get("Nombre Ausnom", "")
        if vac_start is None:
            continue
        base_start = vac_start - timedelta(days=365)
        base_end = vac_start - timedelta(days=1)

        dias_vac_mes, fechas_vac, nota_vac = date_counts_for_rule("VACACIONES", area, vac_start, vac_end, period_start, period_end)

        # Devengos pagados en acumulados dentro de la ventana 365.
        emp_acum = acum[(acum["SAP"] == sap) & (acum["Fecha pago"] >= base_start) & (acum["Fecha pago"] <= base_end)].copy()
        emp_acum["Tipo base"] = np.where(emp_acum["Código"].isin(SALARY_ALL_CODES), "Salario/Bono pagado", np.where(emp_acum["Código"].isin(var_codes), "Variable salarial", "No base"))
        emp_base = emp_acum[emp_acum["Tipo base"].isin(["Salario/Bono pagado", "Variable salarial"])].copy()
        if not emp_base.empty:
            for _, r in emp_base.iterrows():
                devengo_rows.append({
                    "SAP": sap, "Nombre": nombre, "Vacación inicio": vac_start, "Base desde": base_start, "Base hasta": base_end,
                    "Fecha pago": r["Fecha pago"], "Código": r["Código"], "Concepto": r["Concepto"],
                    "Tipo base": r["Tipo base"], "Cantidad": r["Cantidad"], "Importe base": r["Importe"],
                })
        base_salario_pagado = emp_base[emp_base["Tipo base"] == "Salario/Bono pagado"]["Importe"].sum() if not emp_base.empty else 0.0
        base_variable = emp_base[emp_base["Tipo base"] == "Variable salarial"]["Importe"].sum() if not emp_base.empty else 0.0

        # Ausentismos del último año que hacen base.
        emp_abs = aus[(aus["SAP"] == sap) & (aus["Familia ausencia"].isin(abs_families)) & aus.apply(lambda r: overlap(r["Fecha inicio"], r["Fecha fin"], base_start, base_end), axis=1)].copy()
        valor_aus_total = 0.0
        dias_aus_total = 0.0
        for aidx, a in emp_abs.iterrows():
            fam = a["Familia ausencia"]
            abs_start = a["Fecha inicio"]
            abs_end = a["Fecha fin"]
            dias_abs, fechas_abs, nota_abs = date_counts_for_rule(fam, area, abs_start, abs_end, base_start, base_end)
            valor_abs = 0.0
            fuentes = []
            detalles = []
            for d in fechas_abs:
                sal = get_salary_for_date(acum, hist, sap, d, area)
                valor_abs += float(sal["valor_dia_pt"] or 0)
                fuentes.append(str(sal["fuente"]))
                detalles.append(str(sal["detalle"]))
                salario_trace_rows.append({
                    "SAP": sap, "Nombre": nombre, "Fecha valorizada": d, "Área": area,
                    "Salario total usado": sal["salario_total"], "Valor día PT": sal["valor_dia_pt"],
                    "Fuente salario": sal["fuente"], "Detalle salario": sal["detalle"],
                    "Ausentismo": a["Ausentismo SAP"], "Familia": fam,
                })
            valor_aus_total += valor_abs
            dias_aus_total += dias_abs
            ausencia_rows.append({
                "SAP": sap, "Nombre": nombre, "Vacación inicio": vac_start,
                "Base desde": base_start, "Base hasta": base_end,
                "Ausentismo": a["Ausentismo SAP"], "Familia": fam,
                "Inicio ausencia": abs_start, "Fin ausencia": abs_end,
                "Días SAP": a.get("Días SAP", 0), "Días reales promedio": dias_abs,
                "Valor ausentismo promedio": valor_abs,
                "Fuente salario": " | ".join(sorted(set(fuentes))) if fuentes else "",
                "Fechas valorizadas": ", ".join([x.strftime("%d/%m/%Y") for x in fechas_abs[:25]]) + ("..." if len(fechas_abs) > 25 else ""),
                "Observación": nota_abs,
            })

        base_365 = base_salario_pagado + base_variable + valor_aus_total
        valor_diario_vac = base_365 / 365 if base_365 else 0.0
        valor_vac_esperado = valor_diario_vac * dias_vac_mes
        obs = []
        if area not in ["ZH", "ZP"]:
            obs.append("Área no identificada como ZH/ZP")
        if base_365 <= 0:
            obs.append("Base 365 en cero; revisar acumulados/histórico")
        if dias_vac_mes <= 0:
            obs.append("Vacación sin días calculados en el mes revisado")
        if nota_vac:
            obs.append(nota_vac)
        base_rows.append({
            "SAP": sap, "Nombre": nombre, "Área": area,
            "Vacación inicio": vac_start, "Vacación fin": vac_end,
            "Base desde": base_start, "Base hasta": base_end,
            "Días vacaciones SAP": vac.get("Días SAP", 0),
            "Días vacaciones calculados mes": dias_vac_mes,
            "Base salario/bono pagado": base_salario_pagado,
            "Base variables usuario": base_variable,
            "Días ausentismos promedio": dias_aus_total,
            "Base ausentismos valorizados": valor_aus_total,
            "Base 365": base_365,
            "Promedio mensual": base_365 / 365 * 30 if base_365 else 0.0,
            "Valor diario vacaciones": valor_diario_vac,
            "Valor vacaciones esperado": valor_vac_esperado,
            "Estado": "REVISAR" if obs else "OK",
            "Observación": "; ".join(obs) if obs else "OK",
        })

    base_df = pd.DataFrame(base_rows)
    dev_df = pd.DataFrame(devengo_rows)
    aus_df = pd.DataFrame(ausencia_rows)
    sal_df = pd.DataFrame(salario_trace_rows)
    return base_df, dev_df, aus_df, sal_df

# =============================
# Validación mensual y comparación SAP
# =============================
def prepare_monthly_absences(aus: pd.DataFrame, md: pd.DataFrame, year: int, month: int, tolerance_days: float) -> pd.DataFrame:
    period_start = date(year, month, 1)
    period_end = date(year, month, calendar.monthrange(year, month)[1])
    out = aus[aus.apply(lambda r: overlap(r["Fecha inicio"], r["Fecha fin"], period_start, period_end), axis=1)].copy()
    if md is not None and not md.empty:
        out = out.merge(md[["SAP", "Grupo área"]].rename(columns={"Grupo área":"Grupo área MD"}), on="SAP", how="left")
        out["Grupo área"] = np.where(out["Grupo área"].eq("NO_IDENTIFICADA"), out["Grupo área MD"].fillna("NO_IDENTIFICADA"), out["Grupo área"])
    calc = out.apply(lambda r: date_counts_for_rule(r["Familia ausencia"], r["Grupo área"], r["Fecha inicio"], r["Fecha fin"], period_start, period_end), axis=1)
    out["Días calculados app"] = [x[0] for x in calc]
    out["Fechas contadas app"] = [", ".join([d.strftime("%d/%m/%Y") for d in x[1][:30]]) + ("..." if len(x[1]) > 30 else "") for x in calc]
    out["Nota cálculo"] = [x[2] for x in calc]
    out["Diferencia días"] = out["Días SAP"] - out["Días calculados app"]
    out["Estado"] = np.where(out["Diferencia días"].abs() <= tolerance_days, "OK", "REVISAR")
    out["Observación"] = np.where(out["Estado"].eq("OK"), "OK", "Diferencia entre días SAP y días calculados por la app")
    return out


def sum_preno_codes(preno: pd.DataFrame, codes: List[str], value_name: str) -> pd.DataFrame:
    code_set = set([c.strip().upper() for c in codes if c.strip()])
    tmp = preno[preno["Código"].isin(code_set)].copy()
    if tmp.empty:
        return pd.DataFrame(columns=["SAP", value_name])
    return tmp.groupby("SAP", as_index=False)["Valor SAP"].sum().rename(columns={"Valor SAP": value_name})


def compare_vacations_with_preno(base_vac: pd.DataFrame, preno: pd.DataFrame, vac_codes: List[str], tolerance_money: float) -> pd.DataFrame:
    if base_vac is None or base_vac.empty:
        return pd.DataFrame(columns=["SAP", "Valor vacaciones esperado", "Vacaciones SAP", "Diferencia", "Estado", "Observación"])
    expected = base_vac.groupby(["SAP", "Nombre", "Área"], as_index=False).agg({
        "Días vacaciones calculados mes": "sum",
        "Valor vacaciones esperado": "sum",
        "Base 365": "sum",
    })
    sap_vac = sum_preno_codes(preno, vac_codes, "Vacaciones SAP")
    out = expected.merge(sap_vac, on="SAP", how="left")
    out["Vacaciones SAP"] = out["Vacaciones SAP"].fillna(0.0)
    out["Diferencia"] = out["Vacaciones SAP"] - out["Valor vacaciones esperado"]
    out["Estado"] = np.where(out["Diferencia"].abs() <= tolerance_money, "OK", "REVISAR")
    out["Observación"] = np.where(out["Estado"].eq("OK"), "OK", "Diferencia en pago de vacaciones contra base calculada")
    return out


def calculate_payable_days(md: pd.DataFrame, aus_month: pd.DataFrame) -> pd.DataFrame:
    base = md[["SAP", "Nombre MD", "Grupo área"]].copy() if md is not None and not md.empty else pd.DataFrame(columns=["SAP", "Nombre MD", "Grupo área"])
    if aus_month is None or aus_month.empty:
        base["Días descuento ausencias"] = 0.0
        base["Días reconocidos para constantes"] = DIAS_BASE
        return base
    disc = aus_month[aus_month["Familia ausencia"].isin(["ANJ", "LNR", "SUSPENSION"])].groupby("SAP", as_index=False)["Días calculados app"].sum().rename(columns={"Días calculados app":"Días descuento ausencias"})
    out = base.merge(disc, on="SAP", how="left")
    out["Días descuento ausencias"] = out["Días descuento ausencias"].fillna(0.0)
    out["Días reconocidos para constantes"] = (DIAS_BASE - out["Días descuento ausencias"]).clip(0, DIAS_BASE)
    return out


def validate_constants(preno: pd.DataFrame, md: pd.DataFrame, aus_month: pd.DataFrame, aux_value: float, bigpass_value: float, tolerance_money: float, aux_codes: List[str], big_codes: List[str], big_ded_codes: List[str]) -> pd.DataFrame:
    out = calculate_payable_days(md, aus_month)
    for df in [sum_preno_codes(preno, aux_codes, "Auxilio SAP"), sum_preno_codes(preno, big_codes, "BigPass SAP"), sum_preno_codes(preno, big_ded_codes, "Descuento BigPass SAP")]:
        out = out.merge(df, on="SAP", how="left")
    for c in ["Auxilio SAP", "BigPass SAP", "Descuento BigPass SAP"]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = out[c].fillna(0.0)
    out["Auxilio esperado"] = aux_value / DIAS_BASE * out["Días reconocidos para constantes"]
    out["BigPass esperado"] = bigpass_value / DIAS_BASE * out["Días reconocidos para constantes"]
    out["Diferencia auxilio"] = out["Auxilio SAP"] - out["Auxilio esperado"]
    out["Diferencia BigPass"] = out["BigPass SAP"] - out["BigPass esperado"]
    out["Estado"] = np.where((out["Diferencia auxilio"].abs() <= tolerance_money) & (out["Diferencia BigPass"].abs() <= tolerance_money), "OK", "REVISAR")
    out["Observación"] = out.apply(lambda r: "OK" if r["Estado"] == "OK" else "Diferencia en auxilio/BigPass según días reconocidos", axis=1)
    return out

# =============================
# Exportación
# =============================
def auto_width(writer, sheet_name: str, df: pd.DataFrame):
    ws = writer.sheets[sheet_name]
    for i, col in enumerate(df.columns):
        vals = df[col].head(500).fillna("").astype(str).tolist() if col in df.columns else []
        max_len = max([len(str(col))] + [len(x) for x in vals])
        ws.set_column(i, i, min(max_len + 2, 55))


def build_excel_output(sheets: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter", datetime_format="dd/mm/yyyy", date_format="dd/mm/yyyy") as writer:
        for name, df in sheets.items():
            safe_name = name[:31]
            df2 = df.copy() if df is not None else pd.DataFrame()
            for c in df2.columns:
                if df2[c].map(lambda x: isinstance(x, date)).any():
                    df2[c] = pd.to_datetime(df2[c], errors="coerce")
            df2.to_excel(writer, sheet_name=safe_name, index=False)
            auto_width(writer, safe_name, df2)
        wb = writer.book
        header = wb.add_format({"bold": True, "bg_color": "#F28C28", "font_color": "white", "border": 1})
        ok = wb.add_format({"bg_color": "#E7F6E7", "font_color": "#1B5E20"})
        warn = wb.add_format({"bg_color": "#FFF3CD", "font_color": "#8A5A00"})
        for name, df in sheets.items():
            safe_name = name[:31]
            ws = writer.sheets[safe_name]
            for j, col in enumerate(df.columns if df is not None else []):
                ws.write(0, j, col, header)
            if df is not None and "Estado" in df.columns and len(df) > 0:
                idx = list(df.columns).index("Estado")
                ws.conditional_format(1, idx, len(df), idx, {"type": "text", "criteria": "containing", "value": "OK", "format": ok})
                ws.conditional_format(1, idx, len(df), idx, {"type": "text", "criteria": "containing", "value": "REVISAR", "format": warn})
            ws.freeze_panes(1, 0)
            if df is not None and len(df.columns) > 0:
                ws.autofilter(0, 0, max(len(df), 1), len(df.columns) - 1)
    return output.getvalue()

# =============================
# UI
# =============================
def inject_css():
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(135deg, #fffaf2 0%, #f6fff9 48%, #ffffff 100%); }
        .hero { background: linear-gradient(120deg, #F28C28 0%, #00843D 100%); padding: 24px; border-radius: 24px; color: white; box-shadow: 0 10px 28px rgba(0,0,0,.12); }
        .hero h1 { margin: 0; font-size: 2.1rem; }
        .hero p { margin: 6px 0 0 0; font-size: 1rem; opacity: .96; }
        .pt60 { background: #fff3cd; border-left: 7px solid #F28C28; color: #5C3B00; padding: 15px 18px; border-radius: 16px; margin: 18px 0; }
        .card { background: rgba(255,255,255,.88); border: 1px solid rgba(0,132,61,.13); border-radius: 18px; padding: 16px; box-shadow: 0 6px 18px rgba(0,0,0,.06); }
        .footer { text-align:center; color:#49624f; padding-top:24px; font-size:.88rem; }
        div[data-testid="stMetricValue"] { color: #00843D; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def file_sheet_selector(label: str, uploaded_file, key: str) -> Optional[str]:
    if uploaded_file is None or not is_excel_file(uploaded_file):
        return None
    sheets = list_sheets(uploaded_file)
    idx = 0
    low_label = norm_text(label)
    for i, s in enumerate(sheets):
        if "preno" in norm_text(s) and "preno" in low_label:
            idx = i
        if "md" in norm_text(s) and "md" in low_label:
            idx = i
        if "log" not in norm_text(s) and "concept" in norm_text(s) and "concept" in low_label:
            idx = i
    return st.selectbox(label, sheets, index=idx, key=key)


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
    inject_css()
    st.markdown(f"""
    <div class="hero">
        <h1>{APP_ICON} {APP_TITLE}</h1>
        <p>Base de vacaciones 365 días · Comparación contra SAP · Colores JMC · Guacamaya</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="pt60"><b>⚠️ Recordatorio importante:</b><br>
    No olvides correr tiempos de los Part Time a través de la <b>PT60</b> antes de calcular.</div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Parámetros")
        today = date.today()
        year = int(st.number_input("Año de revisión", 2023, 2035, today.year, 1))
        month_name = st.selectbox("Mes de revisión", list(MESES_ES.values()), index=today.month - 1)
        month = list(MESES_ES.values()).index(month_name) + 1
        st.caption("El cálculo de vacaciones usa AUSNOM acumulado y toma 365 días hacia atrás desde el inicio del disfrute.")
        st.divider()
        aux_value = st.number_input("Auxilio transporte mensual", min_value=0.0, value=200000.0, step=1000.0, format="%.0f")
        bigpass_value = st.number_input("BigPass mensual", min_value=0.0, value=0.0, step=1000.0, format="%.0f")
        tolerance_money = st.number_input("Tolerancia diferencias ($)", min_value=0.0, value=100.0, step=50.0, format="%.0f")
        tolerance_days = st.number_input("Tolerancia días", min_value=0.0, value=0.01, step=0.01, format="%.2f")
        st.divider()
        abs_base_families = st.multiselect(
            "Ausentismos que hacen base promedio",
            options=["INCAPACIDAD", "REMUNERADA", "VACACIONES", "OTRA", "ANJ", "LNR", "SUSPENSION"],
            default=DEFAULT_ABS_BASE_FAMILIES,
        )
        with st.expander("🔧 Códigos SAP"):
            vac_codes_txt = st.text_input("Vacaciones", value=", ".join(DEFAULT_VAC_CODES))
            aux_codes_txt = st.text_input("Auxilio", value=", ".join(DEFAULT_AUX_CODES))
            big_codes_txt = st.text_input("BigPass devengo", value=", ".join(DEFAULT_BIGPASS_CODES))
            big_ded_txt = st.text_input("BigPass descuento", value=", ".join(DEFAULT_BIGPASS_DED_CODES))

    st.subheader("1. Carga de archivos")
    a1, a2, a3 = st.columns(3)
    with a1:
        acum_file = st.file_uploader("📚 Acumulados últimos 12 meses", type=["txt", "csv", "xlsx", "xls", "xlsm"], key="acum")
        md_file = st.file_uploader("👥 MD Part Time", type=["txt", "csv", "xlsx", "xls", "xlsm"], key="md")
    with a2:
        aus_acum_file = st.file_uploader("🗓️ AUSNOM acumulado último año", type=["txt", "csv", "xlsx", "xls", "xlsm"], key="aus_acum")
        hist_file = st.file_uploader("📈 Histórico de salarios PT recomendado", type=["txt", "csv", "xlsx", "xls", "xlsm"], key="hist")
    with a3:
        conceptos_file = st.file_uploader("🧩 Matriz conceptos base variable", type=["txt", "csv", "xlsx", "xls", "xlsm"], key="conceptos")
        preno_file = st.file_uploader("📄 Prenómina convertida", type=["xlsx", "xls", "xlsm"], key="preno")
        autoliquidacion_file = st.file_uploader("/662 Base autoliquidación mes anterior (opcional)", type=["txt", "csv", "xlsx", "xls", "xlsm"], key="autol")

    if not all([acum_file, aus_acum_file, conceptos_file, preno_file, md_file]):
        st.info("Carga como mínimo: acumulados, AUSNOM acumulado, matriz de conceptos, prenómina y MD Part Time. El histórico de salarios es recomendado para casos sin salario pagado en acumulados.")
        cal = build_calendar(year, month)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Lunes a viernes", int(cal["Cuenta ZH base L-V"].sum()))
        c2.metric("Sábados", int(cal["Es sábado"].sum()))
        c3.metric("Domingos", int(cal["Es domingo"].sum()))
        c4.metric("Lunes festivos", int(cal["Es lunes festivo"].sum()))
        with st.expander("Ver calendario interno Colombia"):
            st.dataframe(cal, use_container_width=True)
        st.markdown('<div class="footer">Creado por Andrés Huérfano Dávila – Nómina JMC</div>', unsafe_allow_html=True)
        return

    with st.expander("2. Selección de hojas Excel", expanded=False):
        acum_sheet = file_sheet_selector("Hoja acumulados", acum_file, "sh_acum")
        aus_sheet = file_sheet_selector("Hoja AUSNOM acumulado", aus_acum_file, "sh_aus")
        hist_sheet = file_sheet_selector("Hoja histórico salarios", hist_file, "sh_hist") if hist_file else None
        conceptos_sheet = file_sheet_selector("Hoja conceptos base variable", conceptos_file, "sh_con")
        preno_sheet = file_sheet_selector("Hoja prenómina", preno_file, "sh_preno")
        md_sheet = file_sheet_selector("Hoja MD", md_file, "sh_md")
        autol_sheet = file_sheet_selector("Hoja /662", autoliquidacion_file, "sh_autol") if autoliquidacion_file else None

    run = st.button("🚀 Ejecutar motor Part Time", type="primary", use_container_width=True)
    if not run:
        st.markdown('<div class="footer">Creado por Andrés Huérfano Dávila – Nómina JMC</div>', unsafe_allow_html=True)
        return

    try:
        with st.spinner("Leyendo archivos y calculando base de vacaciones 365 días..."):
            raw_md = read_table(md_file, md_sheet)
            md = prepare_md(raw_md)
            raw_acum = read_table(acum_file, acum_sheet)
            acum = prepare_acum(raw_acum)
            raw_aus = read_table(aus_acum_file, aus_sheet)
            aus_acum = prepare_ausnom(raw_aus, md)
            raw_con = read_table(conceptos_file, conceptos_sheet)
            variable_codes = prepare_conceptos_variables(raw_con)
            raw_preno = read_table(preno_file, preno_sheet)
            preno = prepare_preno(raw_preno)
            if hist_file:
                raw_hist = read_table(hist_file, hist_sheet)
                hist = prepare_hist_salarios(raw_hist)
            else:
                hist = pd.DataFrame(columns=["SAP", "Grupo área", "Código", "Importe vigente", "Desde vigencia", "Hasta vigencia"])
            calendar_df = build_calendar(year, month)

            base_vac, detalle_dev, detalle_abs, traza_sal = build_vacation_base(
                acum=acum, aus_acum=aus_acum, hist=hist, md=md, variable_codes=variable_codes,
                year=year, month=month, abs_families_base=abs_base_families,
            )
            aus_month = prepare_monthly_absences(aus_acum, md, year, month, tolerance_days)
            vac_codes = [x.strip().upper() for x in vac_codes_txt.split(",") if x.strip()]
            aux_codes = [x.strip().upper() for x in aux_codes_txt.split(",") if x.strip()]
            big_codes = [x.strip().upper() for x in big_codes_txt.split(",") if x.strip()]
            big_ded_codes = [x.strip().upper() for x in big_ded_txt.split(",") if x.strip()]
            comp_vac = compare_vacations_with_preno(base_vac, preno, vac_codes, tolerance_money)
            comp_const = validate_constants(preno, md, aus_month, aux_value, bigpass_value, tolerance_money, aux_codes, big_codes, big_ded_codes)

            log_rows = [
                {"Tipo": "OK", "Detalle": "Proceso ejecutado correctamente"},
                {"Tipo": "Parámetro", "Detalle": f"Período de revisión: {MESES_ES[month]} {year}"},
                {"Tipo": "Parámetro", "Detalle": "AUSNOM acumulado se evaluó con ventana 365 días desde inicio de vacaciones"},
                {"Tipo": "Parámetro", "Detalle": f"Familias de ausentismo que hacen base: {', '.join(abs_base_families)}"},
                {"Tipo": "Parámetro", "Detalle": f"Conceptos variables cargados: {', '.join(variable_codes) if variable_codes else 'Ninguno'}"},
            ]
            if hist.empty:
                log_rows.append({"Tipo": "Advertencia", "Detalle": "No se cargó histórico de salarios PT; se usará acumulados y último acumulado anterior como respaldo."})
            if autoliquidacion_file:
                log_rows.append({"Tipo": "Info", "Detalle": "/662 cargado. Queda reservado para fase de incapacidades/IBC."})
            log_df = pd.DataFrame(log_rows)

            resumen = pd.DataFrame([
                {"Indicador": "Empleados MD Part Time", "Valor": len(md)},
                {"Indicador": "Registros acumulados", "Valor": len(acum)},
                {"Indicador": "Registros AUSNOM acumulado", "Valor": len(aus_acum)},
                {"Indicador": "Vacaciones detectadas en mes", "Valor": len(base_vac)},
                {"Indicador": "Vacaciones OK", "Valor": int((comp_vac.get("Estado", pd.Series(dtype=str)) == "OK").sum()) if not comp_vac.empty else 0},
                {"Indicador": "Vacaciones a revisar", "Valor": int((comp_vac.get("Estado", pd.Series(dtype=str)) == "REVISAR").sum()) if not comp_vac.empty else 0},
                {"Indicador": "Ausentismos mes a revisar", "Valor": int((aus_month.get("Estado", pd.Series(dtype=str)) == "REVISAR").sum()) if not aus_month.empty else 0},
                {"Indicador": "Constantes a revisar", "Valor": int((comp_const.get("Estado", pd.Series(dtype=str)) == "REVISAR").sum()) if not comp_const.empty else 0},
                {"Indicador": "Histórico salarios cargado", "Valor": "Sí" if not hist.empty else "No"},
                {"Indicador": "Días base constante", "Valor": DIAS_BASE},
            ])

            sheets = {
                "Resumen": resumen,
                "Base_Vacaciones": base_vac,
                "Comparacion_Vacaciones_SAP": comp_vac,
                "Detalle_Devengos_Base": detalle_dev,
                "Detalle_Ausentismos_365": detalle_abs,
                "Traza_Salario_Ausencias": traza_sal,
                "Ausentismos_Mes_Validados": aus_month,
                "Constantes_Aux_BigPass": comp_const,
                "Calendario_App": calendar_df,
                "Conceptos_Base_Variable": pd.DataFrame({"Código": variable_codes}),
                "MD_PT_Normalizado": md,
                "Acumulados_Normalizados": acum,
                "AUSNOM_Acum_Normalizado": aus_acum,
                "Hist_Salarios_Normalizado": hist,
                "Prenomina_Normalizada": preno,
                "Log_Proceso": log_df,
            }
            excel_bytes = build_excel_output(sheets)

        st.success("Proceso finalizado. La base de vacaciones ya usa AUSNOM acumulado y ventana de 365 días.")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Vacaciones detectadas", len(base_vac))
        k2.metric("Vacaciones a revisar", int((comp_vac["Estado"] == "REVISAR").sum()) if not comp_vac.empty else 0)
        k3.metric("Ausentismos mes a revisar", int((aus_month["Estado"] == "REVISAR").sum()) if not aus_month.empty else 0)
        k4.metric("Registros AUSNOM acumulado", len(aus_acum))

        t1, t2, t3, t4, t5 = st.tabs(["Resumen", "Base vacaciones", "Vacaciones vs SAP", "Ausentismos 365", "Constantes"])
        with t1:
            st.dataframe(resumen, use_container_width=True)
        with t2:
            st.dataframe(base_vac, use_container_width=True)
        with t3:
            st.dataframe(comp_vac, use_container_width=True)
        with t4:
            st.dataframe(detalle_abs, use_container_width=True)
        with t5:
            st.dataframe(comp_const, use_container_width=True)

        filename = f"revision_part_time_vacaciones_{year}_{month:02d}.xlsx"
        st.download_button("⬇️ Descargar Excel de revisión", data=excel_bytes, file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

    except Exception as e:
        st.error("No fue posible completar el proceso.")
        st.exception(e)
        st.info("Revisa encabezados, formato SAP con pipes y que el AUSNOM sea acumulado del último año, no solo el mes.")

    st.markdown('<div class="footer">Creado por Andrés Huérfano Dávila – Nómina JMC</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
