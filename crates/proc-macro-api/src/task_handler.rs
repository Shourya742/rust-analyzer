//! Task handling abstraction
use crate::ServerError;

pub trait TaskHandler {
    type Request;
    type Response;

    fn send_task(&mut self, task: Self::Request) -> Result<Self::Response, ServerError>;
}
