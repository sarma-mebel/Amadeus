/** Local-date stamp (YYYY-MM-DD). The backend stamps brief/diary/rhythm
 * dates from the server's LOCAL date, so every client-side "is this today?"
 * comparison must build the same local stamp — `toISOString().slice(0, 10)`
 * is the UTC date, which in UTC+ timezones lags the local one all morning
 * (hiding a same-day brief/diary card exactly when it matters). */
export const localDate = (nowMs: number = Date.now()): string => {
  const d = new Date(nowMs)
  const month = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${d.getFullYear()}-${month}-${day}`
}
