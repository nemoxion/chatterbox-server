#![windows_subsystem = "windows"]

use serde::Deserialize;
use std::{
    cell::RefCell,
    fs::{self, OpenOptions},
    io::{Read, Write},
    mem::{size_of, zeroed},
    net::{SocketAddr, TcpStream},
    os::windows::{ffi::OsStrExt, io::AsRawHandle, process::CommandExt},
    path::PathBuf,
    process::{Child, Command, Stdio},
    ptr::{null, null_mut},
    time::Duration,
};
use windows_sys::Win32::{
    Foundation::*,
    System::{JobObjects::*, LibraryLoader::GetModuleHandleW, Threading::*},
    UI::{Shell::*, WindowsAndMessaging::*},
};

const CALLBACK: u32 = WM_APP + 1;
const CONTROL: u32 = WM_APP + 2;
const OPEN: u32 = 1;
const DOCS: u32 = 2;
const START: u32 = 3;
const STOP: u32 = 4;
const LOGS: u32 = 5;
const EXIT: u32 = 6;
const UNLOAD: u32 = 7;
const CUSTOM_TIMEOUT: u32 = 8;
const TIMEOUTS: [u64; 6] = [0, 1800, 3600, 7200, 14400, 28800];

fn server_request(method: &str, path: &str, body: &str) -> Result<serde_json::Value, String> {
    let mut stream = TcpStream::connect_timeout(
        &"127.0.0.1:7860".parse::<SocketAddr>().unwrap(),
        Duration::from_secs(2),
    )
    .map_err(|e| e.to_string())?;
    stream
        .set_read_timeout(Some(Duration::from_secs(if method == "GET" {
            2
        } else {
            180
        })))
        .map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .map_err(|e| e.to_string())?;
    let request = format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
    stream
        .write_all(request.as_bytes())
        .map_err(|e| e.to_string())?;
    let mut response = String::new();
    stream
        .take(65536)
        .read_to_string(&mut response)
        .map_err(|e| e.to_string())?;
    let (headers, body) = response
        .split_once("\r\n\r\n")
        .ok_or("Invalid server response")?;
    if !headers.starts_with("HTTP/1.1 200") {
        return Err(format!("Server request failed: {body}"));
    }
    serde_json::from_str(body).map_err(|e| e.to_string())
}

fn background_request(method: &'static str, path: &'static str, body: String) {
    std::thread::spawn(move || {
        if let Err(error) = server_request(method, path, &body) {
            unsafe {
                message(&error);
            }
        }
    });
}

#[derive(Deserialize)]
struct Config {
    repository: PathBuf,
    python: PathBuf,
}

struct App {
    config: Config,
    child: Option<Child>,
    job: HANDLE,
    icon: HICON,
    taskbar_message: u32,
}

fn wide(value: impl AsRef<std::ffi::OsStr>) -> Vec<u16> {
    value.as_ref().encode_wide().chain(Some(0)).collect()
}

fn server_running() -> bool {
    let address: SocketAddr = "127.0.0.1:7860".parse().unwrap();
    let Ok(mut stream) = TcpStream::connect_timeout(&address, Duration::from_millis(200)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(500)));
    if stream
        .write_all(b"GET /model/status HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .is_err()
    {
        return false;
    }
    let mut data = String::new();
    let _ = stream.take(8192).read_to_string(&mut data);
    data.starts_with("HTTP/1.1 200") && data.contains("chatterbox-multilingual-v3")
}

unsafe fn message(text: &str) {
    MessageBoxW(
        null_mut(),
        wide(text).as_ptr(),
        wide("Chatterbox Server").as_ptr(),
        MB_OK | MB_ICONERROR,
    );
}

unsafe fn open(target: impl AsRef<std::ffi::OsStr>) {
    ShellExecuteW(
        null_mut(),
        wide("open").as_ptr(),
        wide(target).as_ptr(),
        null(),
        null(),
        SW_SHOWNORMAL,
    );
}

impl App {
    fn start(&mut self) -> Result<(), String> {
        if self.child.is_some() || server_running() {
            return Ok(());
        }
        // Never replace a different service on the same port.
        if TcpStream::connect("127.0.0.1:7860").is_ok() {
            return Err("Port 7860 is occupied by another service.".into());
        }
        let runtime = self.config.repository.join("runtime");
        fs::create_dir_all(&runtime).map_err(|e| e.to_string())?;
        let stdout = OpenOptions::new()
            .create(true)
            .append(true)
            .open(runtime.join("server.stdout.log"))
            .map_err(|e| e.to_string())?;
        let stderr = OpenOptions::new()
            .create(true)
            .append(true)
            .open(runtime.join("server.stderr.log"))
            .map_err(|e| e.to_string())?;
        let mut child = Command::new(&self.config.python)
            .arg(self.config.repository.join("local_app.py"))
            .current_dir(&self.config.repository)
            .env("PYTHONUNBUFFERED", "1")
            .creation_flags(CREATE_NO_WINDOW)
            .stdin(Stdio::null())
            .stdout(stdout)
            .stderr(stderr)
            .spawn()
            .map_err(|e| format!("Cannot start Python: {e}"))?;
        // The job contains both the venv launcher and the real Python process.
        unsafe {
            if AssignProcessToJobObject(self.job, child.as_raw_handle() as HANDLE) == 0 {
                let error = std::io::Error::last_os_error();
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("Cannot manage server process: {error}"));
            }
        }
        self.child = Some(child);
        Ok(())
    }

    fn stop(&mut self) {
        if self.child.is_some() {
            unsafe {
                TerminateJobObject(self.job, 0);
            }
            if let Some(mut child) = self.child.take() {
                let _ = child.wait();
            }
        }
    }

    fn reap(&mut self) {
        let exited = self
            .child
            .as_mut()
            .is_some_and(|child| child.try_wait().ok().flatten().is_some());
        if exited {
            self.stop();
        }
    }

    unsafe fn notify(&self, window: HWND, action: u32, tooltip: &str) {
        let mut data: NOTIFYICONDATAW = zeroed();
        data.cbSize = size_of::<NOTIFYICONDATAW>() as u32;
        data.hWnd = window;
        data.uID = 1;
        data.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP;
        data.uCallbackMessage = CALLBACK;
        data.hIcon = self.icon;
        let text = wide(tooltip);
        let count = text.len().min(data.szTip.len() - 1);
        data.szTip[..count].copy_from_slice(&text[..count]);
        Shell_NotifyIconW(action, &data);
    }

    unsafe fn menu(&mut self, window: HWND) {
        self.reap();
        let running = server_running();
        let owned = self.child.is_some();
        let label = if running && owned {
            "Server running"
        } else if running {
            "Server running (started elsewhere)"
        } else if owned {
            "Server starting - see logs if it does not become ready"
        } else {
            "Server stopped"
        };
        let menu = CreatePopupMenu();
        AppendMenuW(menu, MF_STRING | MF_GRAYED, 0, wide(label).as_ptr());
        AppendMenuW(menu, MF_SEPARATOR, 0, null());
        for (id, text, enabled) in [
            (OPEN, "Open Chatterbox", true),
            (DOCS, "Open API documentation", true),
            (START, "Start server", !owned && !running),
            (STOP, "Stop server", owned),
            (UNLOAD, "Unload model (keep server running)", running),
            (LOGS, "Open logs folder", true),
            (EXIT, "Quit Chatterbox (stops managed server)", true),
        ] {
            AppendMenuW(
                menu,
                MF_STRING | if enabled { 0 } else { MF_GRAYED },
                id as usize,
                wide(text).as_ptr(),
            );
        }
        let timeout_menu = CreatePopupMenu();
        let current = if running {
            server_request("GET", "/model/settings", "")
                .ok()
                .and_then(|v| v["idle_timeout_seconds"].as_u64())
        } else {
            None
        };
        for (index, (seconds, label)) in TIMEOUTS
            .iter()
            .zip([
                "Disabled",
                "30 minutes",
                "1 hour",
                "2 hours (default)",
                "4 hours",
                "8 hours",
            ])
            .enumerate()
        {
            let flags = MF_STRING
                | if current == Some(*seconds) {
                    MF_CHECKED
                } else {
                    0
                }
                | if running { 0 } else { MF_GRAYED };
            AppendMenuW(timeout_menu, flags, 100 + index, wide(label).as_ptr());
        }
        AppendMenuW(
            timeout_menu,
            MF_STRING,
            CUSTOM_TIMEOUT as usize,
            wide("Custom timeout (web interface)").as_ptr(),
        );
        AppendMenuW(
            menu,
            MF_POPUP,
            timeout_menu as usize,
            wide("Auto-unload idle model").as_ptr(),
        );
        let mut point: POINT = zeroed();
        GetCursorPos(&mut point);
        SetForegroundWindow(window);
        let chosen = TrackPopupMenu(
            menu,
            TPM_RETURNCMD | TPM_RIGHTBUTTON,
            point.x,
            point.y,
            0,
            window,
            null(),
        ) as u32;
        DestroyMenu(menu);
        PostMessageW(window, WM_NULL, 0, 0);
        self.action(window, chosen);
    }

    unsafe fn action(&mut self, window: HWND, chosen: u32) {
        match chosen {
            OPEN => open("http://127.0.0.1:7860/"),
            DOCS => open("http://127.0.0.1:7860/docs"),
            START => {
                if let Err(error) = self.start() {
                    message(&error);
                }
            }
            STOP => self.stop(),
            UNLOAD => background_request("POST", "/model/unload", String::new()),
            CUSTOM_TIMEOUT => open("http://127.0.0.1:7860/"),
            id @ 100..=105 => {
                let seconds = TIMEOUTS[(id - 100) as usize];
                background_request(
                    "PUT",
                    "/model/settings",
                    format!("{{\"idle_timeout_seconds\":{seconds}}}"),
                );
            }
            LOGS => open(self.config.repository.join("runtime")),
            EXIT => {
                PostMessageW(window, WM_CLOSE, 0, 0);
            }
            _ => {}
        }
    }
}

unsafe extern "system" fn window_proc(
    window: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // DestroyWindow synchronously sends WM_DESTROY: do not hold a state borrow here.
    if msg == WM_CLOSE {
        DestroyWindow(window);
        return 0;
    }
    let ptr = GetWindowLongPtrW(window, GWLP_USERDATA) as *mut RefCell<App>;
    if !ptr.is_null() {
        // Menus run a nested message loop; skip callbacks while state is borrowed.
        let Ok(mut app) = (*ptr).try_borrow_mut() else {
            return DefWindowProcW(window, msg, wparam, lparam);
        };
        if msg == app.taskbar_message {
            app.notify(window, NIM_ADD, "Chatterbox Server");
            return 0;
        }
        match msg {
            CONTROL => {
                app.action(window, wparam as u32);
                return 0;
            }
            CALLBACK => {
                match lparam as u32 {
                    WM_RBUTTONUP | WM_CONTEXTMENU => app.menu(window),
                    WM_LBUTTONDBLCLK => open("http://127.0.0.1:7860/"),
                    _ => {}
                }
                return 0;
            }
            WM_TIMER => {
                app.reap();
                let tip = if server_running() {
                    "Chatterbox Server - running"
                } else if app.child.is_some() {
                    "Chatterbox Server - starting"
                } else {
                    "Chatterbox Server - stopped"
                };
                app.notify(window, NIM_MODIFY, tip);
                return 0;
            }
            WM_DESTROY => {
                app.stop();
                app.notify(window, NIM_DELETE, "");
                KillTimer(window, 1);
                PostQuitMessage(0);
                return 0;
            }
            _ => {}
        }
    }
    DefWindowProcW(window, msg, wparam, lparam)
}

fn main() {
    if let Err(error) = run() {
        unsafe {
            message(&error);
        }
    }
}

fn run() -> Result<(), String> {
    if let Some(command) = std::env::args().nth(1) {
        let id = match command.as_str() {
            "--start-server" => START,
            "--stop-server" => STOP,
            "--quit" => EXIT,
            "--unload-model" => UNLOAD,
            "--timeout-disabled" => 100,
            "--timeout-30m" => 101,
            "--timeout-1h" => 102,
            "--timeout-2h" => 103,
            "--timeout-4h" => 104,
            "--timeout-8h" => 105,
            _ => return Err("Supported commands: --start-server, --stop-server, --quit".into()),
        };
        unsafe {
            let window = FindWindowW(wide("ChatterboxTrayWindow").as_ptr(), null());
            if window.is_null() {
                return Err("The Chatterbox tray application is not running.".into());
            }
            PostMessageW(window, CONTROL, id as usize, 0);
        }
        return Ok(());
    }
    let executable = std::env::current_exe().map_err(|e| e.to_string())?;
    let folder = executable.parent().ok_or("Missing application directory")?;
    let config_path = folder.join("config.json");
    let config: Config = serde_json::from_slice(
        &fs::read(&config_path)
            .map_err(|e| format!("Cannot read {}: {e}", config_path.display()))?,
    )
    .map_err(|e| e.to_string())?;
    if !config.python.is_file() || !config.repository.join("local_app.py").is_file() {
        return Err("Python or server source is missing. Reinstall with the correct repository and Python paths.".into());
    }
    unsafe {
        let mutex = CreateMutexW(null(), 0, wide("Local\\ChatterboxServerTray").as_ptr());
        if mutex.is_null() {
            return Err(std::io::Error::last_os_error().to_string());
        }
        if GetLastError() == ERROR_ALREADY_EXISTS {
            CloseHandle(mutex);
            return Ok(());
        }
        let job = CreateJobObjectW(null(), null());
        if job.is_null() {
            return Err("Cannot create a server process group".into());
        }
        let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = zeroed();
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &limits as *const _ as _,
            size_of_val(&limits) as u32,
        ) == 0
        {
            CloseHandle(job);
            return Err("Cannot configure server process group".into());
        }
        let instance = GetModuleHandleW(null());
        let class_name = wide("ChatterboxTrayWindow");
        let class = WNDCLASSW {
            lpfnWndProc: Some(window_proc),
            hInstance: instance,
            lpszClassName: class_name.as_ptr(),
            ..zeroed()
        };
        if RegisterClassW(&class) == 0 {
            return Err("Cannot register tray window".into());
        }
        let window = CreateWindowExW(
            0,
            class_name.as_ptr(),
            wide("Chatterbox Server").as_ptr(),
            0,
            0,
            0,
            0,
            0,
            null_mut(),
            null_mut(),
            instance,
            null(),
        );
        if window.is_null() {
            return Err("Cannot create tray window".into());
        }
        let custom_icon = LoadImageW(
            null_mut(),
            wide(folder.join("app.ico")).as_ptr(),
            IMAGE_ICON,
            32,
            32,
            LR_LOADFROMFILE,
        ) as HICON;
        let icon = if custom_icon.is_null() {
            LoadIconW(null_mut(), IDI_APPLICATION)
        } else {
            custom_icon
        };
        let app = Box::new(RefCell::new(App {
            config,
            child: None,
            job,
            icon,
            taskbar_message: RegisterWindowMessageW(wide("TaskbarCreated").as_ptr()),
        }));
        SetWindowLongPtrW(
            window,
            GWLP_USERDATA,
            (&*app) as *const RefCell<App> as isize,
        );
        app.borrow()
            .notify(window, NIM_ADD, "Chatterbox Server - starting");
        SetTimer(window, 1, 3000, None);
        let result = app.borrow_mut().start();
        if let Err(error) = result {
            message(&error);
        }
        let mut msg: MSG = zeroed();
        while GetMessageW(&mut msg, null_mut(), 0, 0) > 0 {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        SetWindowLongPtrW(window, GWLP_USERDATA, 0);
        app.borrow_mut().stop();
        CloseHandle(job);
        CloseHandle(mutex);
        if !custom_icon.is_null() {
            DestroyIcon(custom_icon);
        }
    }
    Ok(())
}
