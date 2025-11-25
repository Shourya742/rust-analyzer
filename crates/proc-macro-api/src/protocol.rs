//! Entry point of protocol supported by proc macro
use std::io::{self, BufRead, Write};

use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::Codec;

pub mod bidirectional;
pub mod legacy;

#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ServerConfig {
    pub span_mode: SpanMode,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PanicMessage(pub String);

/// Span Mode
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SpanMode {
    #[default]
    Id,
    RustAnalyzer,
}

pub trait Message: serde::Serialize + DeserializeOwned {
    fn read<R: BufRead, C: Codec>(inp: &mut R, buf: &mut C::Buf) -> io::Result<Option<Self>> {
        Ok(match C::read(inp, buf)? {
            None => None,
            Some(buf) => C::decode(buf)?,
        })
    }
    fn write<W: Write, C: Codec>(self, out: &mut W) -> io::Result<()> {
        let value = C::encode(&self)?;
        C::write(out, &value)
    }
}
