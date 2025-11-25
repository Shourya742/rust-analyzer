//! Message definition for bidirectional protocol.
use paths::Utf8PathBuf;
use serde::{Deserialize, Serialize};

use crate::{
    ProcMacroKind,
    protocol::{Message, PanicMessage, ServerConfig},
    transport::flat::FlatTree,
};

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

impl From<Response> for ServerMessage {
    fn from(value: Response) -> Self {
        ServerMessage::Response(value)
    }
}

impl From<Request> for ClientMessage {
    fn from(value: Request) -> Self {
        ClientMessage::Request(value)
    }
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

impl Message for ClientMessage {}
impl Message for ServerMessage {}
