// Crear una función LeerExportacion con este código.
// Tres consultas: LeerExportacion(RutaApp, "summary"), "weekly", "inventory".
(RutaApp as text, Tabla as text) as table =>
let
    Base = Text.TrimEnd(RutaApp, {"\", "/"}) & "\exports\",
    Puntero = Json.Document(File.Contents(Base & "latest.json")),
    Archivo = Base & Puntero[run_id] & "\" & Tabla & ".csv",
    Permitida = List.Contains({"summary", "weekly", "inventory"}, Tabla),
    Datos = if Permitida then Csv.Document(File.Contents(Archivo), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]) else error "Tabla no permitida",
    Cabeceras = Table.PromoteHeaders(Datos, [PromoteAllScalars=true])
in
    Cabeceras
