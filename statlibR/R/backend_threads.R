backend_thread_count <- function() {
  .Call(
    C_backend_thread_count,
    ensure_backend_available()
  )
}
