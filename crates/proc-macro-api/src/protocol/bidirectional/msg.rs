use paths::Utf8PathBuf;
use serde::{Deserialize, Serialize};

use crate::{ProcMacroKind, transport::flat::FlatTree};

#[derive(Debug, Serialize, Deserialize)]
pub enum ClientMessage {
    Request(Request),
    Reply,
    Prompt,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Request {
    ListMacros { dylib_path: Utf8PathBuf },
    ExpandMacro(Box<ExpandMacro>),
    ApiVersionCheck {},
    SetConfig(ServerConfig),
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ServerMessage {
    Response(Response),
    Prompt,
    Reply,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Response {
    ListMacros(Result<Vec<(String, ProcMacroKind)>, String>),
    ExpandMacro(Result<FlatTree, PanicMessage>),
    ApiVersionCheck(u32),
    SetConfig(ServerConfig),
    ExpandMacroExtended(Result<ExpandMacroExtended, PanicMessage>),
}

#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ServerConfig {
    pub span_mode: SpanMode,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PanicMessage(pub String);

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize)]
pub enum SpanMode {
    #[default]
    Id,
    RustAnalyzer,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExpandMacro {
    pub lib: Utf8PathBuf,
    pub env: Vec<(String, String)>,
    pub current_dir: Option<String>,
    #[serde(flatten)]
    pub data: ExpandMacroData,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExpandMacroExtended {
    pub tree: FlatTree,
    pub span_data_table: Vec<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExpandMacroData {
    pub macro_body: FlatTree,
    pub macro_name: String,
    pub attributes: Option<FlatTree>,
    #[serde(skip_serializing_if = "ExpnGlobals::skip_serializing_if")]
    #[serde(default)]
    pub has_global_spans: ExpnGlobals,

    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(default)]
    pub span_data_table: Vec<u32>,
}

#[derive(Clone, Copy, Default, Debug, Serialize, Deserialize)]
pub struct ExpnGlobals {
    #[serde(skip_serializing)]
    #[serde(default)]
    pub serialize: bool,
    pub def_site: usize,
    pub call_site: usize,
    pub mixed_site: usize,
}

impl ExpnGlobals {
    fn skip_serializing_if(&self) -> bool {
        !self.serialize
    }
}
