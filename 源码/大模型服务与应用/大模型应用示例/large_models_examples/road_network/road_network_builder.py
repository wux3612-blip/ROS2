#!/usr/bin/env python3
import copy
import math
import os
import sys

import cv2
import yaml
from PyQt5.QtCore import QEvent, QPointF, QRectF, Qt, QTimer
from PyQt5.QtGui import (
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPolygonF,
)
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QInputDialog,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
MIN_ZOOM = 0.2
MAX_ZOOM = 40.0
ZOOM_FACTOR = 1.2
PAN_STEP = 60
DRAG_THRESHOLD = 5
DEFAULT_SAVE_NAME = "road_network.yaml"
CONFIG_DIR = "/home/ubuntu/ros2_ws/src/large_models_examples/large_models_examples/road_network/config"
MAP_DIR = "/home/ubuntu/ros2_ws/src/slam/maps"


TEXT = {
    "English": {
        "title": "Road Network Editor",
        "mode_add": "Add",
        "mode_edit": "Edit",
        "mode_pan": "Move View",
        "mode_select": "Select",
        "add_node": "Add Waypoint (A)",
        "add_node_help": "Left click adds a waypoint; drag sets direction; right click connects waypoints.",
        "edit_node": "Edit Waypoint (E)",
        "edit_node_help": "Left drag moves the selected waypoint; drag direction updates its heading.",
        "move_view": "Move View (V)",
        "move_view_help": "Left drag pans the map view without changing waypoints.",
        "move_origin": "Move Origin (O)",
        "fix_node": "Fixed Waypoint (K)",
        "fix_node_help": "Click waypoints to lock or unlock them; fixed waypoints stay in place during smart align.",
        "orthogonal_node": "Orthogonal Waypoint (X)",
        "orthogonal_help": "Orthogonal Waypoint is on: heading snaps to 0, 90, 180, or -90 degrees.",
        "rename_node": "Rename ID",
        "rename_node_help": "Rename ID is on: click a waypoint number, then enter a new ID.",
        "undo": "Undo (Z)",
        "redo": "Redo (Y)",
        "save": "Save (S)",
        "reload": "Reload (L)",
        "align": "Smart Align (G)",
        "align_help": "Smart Align keeps the origin and fixed waypoints in place, then straightens connected waypoints.",
        "delete": "Delete Waypoint (D)",
        "clear_nodes": "Clear Waypoints (C)",
        "zoom_in": "Zoom In (+)",
        "zoom_out": "Zoom Out (-)",
        "fit": "Fit Window (F)",
        "save_quit": "Save & Quit (Q)",
        "quit": "Quit (Esc)",
        "language": "中文",
        "file_label": "Network YAML:",
        "file_placeholder": DEFAULT_SAVE_NAME,
        "file_select": "Select YAML",
        "file_exists": "Loaded {path}",
        "file_missing_title": "File does not exist",
        "file_missing_message": "{path} does not exist.",
        "create_file": "Create File",
        "confirm": "Confirm",
        "cancel": "Cancel",
        "created_file": "Created {path}",
        "save_confirm_title": "Confirm Save",
        "save_confirm_message": "Save the current road network YAML?\n\n{path}",
        "save_confirm_missing_message": "This YAML file does not exist yet. Create and save it?\n\n{path}",
        "hint": (
            "Add Waypoint (A): left click adds, drag sets direction, right click connects waypoints | "
            "Edit Waypoint (E): left drag to modify | "
            "Move View (V): left drag to pan | "
            "Orthogonal Waypoint: snap waypoint heading to cross directions | "
            "Fixed Waypoint (K): click waypoints to lock/unlock | "
            "Clear Waypoints (C): clear non-reference waypoints | "
            "Green line: one-way | Blue line: two-way | "
            "Delete Waypoint: click a waypoint | Move Origin: drag to place origin | "
            "Mouse wheel: zoom | Middle drag / arrow keys: pan"
        ),
        "status": "Mode: {mode}   Tool: {tool}   Zoom: {zoom:.2f}x   Mouse: {mouse}   Selected: {selected}   Pose: {pose}",
        "tool_none": "Normal",
        "tool_delete": "Delete Waypoint",
        "tool_origin": "Move Origin",
        "tool_fix": "Fixed Waypoint",
        "mouse_empty": "-",
        "selected_empty": "-",
        "saved": "Saved to {path}",
        "save_error": "Save error: {error}",
        "reloaded": "Reloaded waypoints",
        "undo_done": "Undo",
        "redo_done": "Redo",
        "align_empty": "No waypoints to align",
        "align_done": "Smart alignment completed",
        "delete_ref": "ID 0 reference point cannot be deleted",
        "fixed_ref": "Fixed waypoint cannot be changed",
        "tool_rename": "Rename ID",
        "rename_prompt": "Rename ID tool active: click a waypoint number",
        "rename_title": "Rename Waypoint ID",
        "rename_label": "New ID:",
        "rename_conflict_title": "ID Conflict",
        "rename_conflict_message": "Waypoint ID {id} already exists. Please enter another ID.",
        "rename_invalid_title": "Invalid ID",
        "rename_invalid_message": "Waypoint ID must be a non-negative integer. ID 0 is reserved for the origin and cannot be renamed.",
        "rename_done": "Renamed waypoint {old_id} to {new_id}",
        "delete_prompt": "Delete tool active: click a waypoint to delete it",
        "delete_done": "Deleted waypoint {id}",
        "edit_delete_done": "Deleted waypoint {id}; left click to reset this ID",
        "reset_deleted": "Reset waypoint {id}: ({x:.2f}, {y:.2f}, {yaw:.2f})",
        "nudged": "Moved waypoint {id}: ({x:.2f}, {y:.2f})",
        "nudge_missing": "Select a waypoint before using arrow-key adjustment",
        "origin_nudge_missing": "Select ID 0 before using arrow-key origin adjustment",
        "clear_empty": "No non-reference waypoints to clear",
        "clear_done": "Cleared all non-reference waypoints",
        "edit_ref": "ID 0 reference point cannot be moved",
        "origin_prompt": "Move origin tool active: click ID 0 to select it; arrow keys move it by 0.01 m",
        "origin_moved": "Moved origin: ({x:.2f}, {y:.2f}, {yaw:.2f})",
        "selected_origin": "Selected origin for editing",
        "fix_prompt": "Fixed waypoint tool active: click waypoints to lock or unlock them",
        "fixed_node": "Fixed waypoint {id}",
        "unfixed_node": "Unlocked waypoint {id}",
        "view_prompt": "Move view mode active: drag with the left mouse button to pan",
        "selected_edit": "Selected waypoint {id} for editing",
        "selected_pose": "Pose: ({x:.2f}, {y:.2f}, {yaw:.2f})",
        "added": "Added waypoint {id}: ({x:.2f}, {y:.2f}, {yaw:.2f})",
        "edited": "Edited waypoint {id}: ({x:.2f}, {y:.2f}, {yaw:.2f})",
        "conn_start": "Selected {id} as connection start",
        "conn_add": "Connection: {src} -> {dst}",
        "conn_remove": "Removed connection: {src} -> {dst}",
        "outside_add": "Ignored waypoint outside map",
        "outside_edit": "Ignored edit outside map",
        "load_error": "Error loading waypoints: {error}",
        "map_load_error": "Unable to load map image: {path}",
        "no_map": "Please specify map YAML file",
    },
    "Chinese": {
        "title": "道路网络编辑器",
        "mode_add": "添加",
        "mode_edit": "修改",
        "mode_pan": "移动视角",
        "mode_select": "查看",
        "add_node": "添加路点 (A)",
        "add_node_help": "鼠标左键按下添加路点，拖动表示方向，右键连接路点",
        "edit_node": "修改路点 (E)",
        "edit_node_help": "鼠标左键拖拽修改路点位置，拖动方向会更新路点朝向",
        "move_view": "移动视角 (V)",
        "move_view_help": "鼠标左键拖动画面，只移动视角，不修改路点",
        "move_origin": "移动原点 (O)",
        "fix_node": "固定路点 (K)",
        "fix_node_help": "点击地图上的路点可固定或解锁；固定路点在智能整理时不会移动",
        "orthogonal_node": "正交路点 (X)",
        "orthogonal_help": "正交路点已开启：路点朝向会吸附到 0、90、180、-90 度",
        "rename_node": "修改序号",
        "rename_node_help": "修改序号已开启：单击地图上的节点序号，然后输入新的 ID",
        "undo": "撤销 (Z)",
        "redo": "重做 (Y)",
        "save": "保存 (S)",
        "reload": "重新加载 (L)",
        "align": "智能整理 (G)",
        "align_help": "智能整理会保持原点和固定路点不动，并整理已连接路点的横平竖直关系",
        "delete": "删除路点 (D)",
        "clear_nodes": "清空路点 (C)",
        "zoom_in": "放大 (+)",
        "zoom_out": "缩小 (-)",
        "fit": "窗口适配 (F)",
        "save_quit": "保存并退出 (Q)",
        "quit": "退出 (Esc)",
        "language": "English",
        "file_label": "路网文件：",
        "file_placeholder": DEFAULT_SAVE_NAME,
        "file_select": "选择 YAML",
        "file_exists": "已加载 {path}",
        "file_missing_title": "文件不存在",
        "file_missing_message": "{path} 不存在。",
        "create_file": "创建文件",
        "confirm": "确认",
        "cancel": "取消",
        "created_file": "已创建 {path}",
        "save_confirm_title": "确认保存",
        "save_confirm_message": "是否保存当前路径下的 YAML？\n\n{path}",
        "save_confirm_missing_message": "当前 YAML 文件不存在，是否创建并保存？\n\n{path}",
        "hint": "添加路点(A)：左键按下添加路点，拖动表示方向，右键连接路点 | 修改路点(E)：左键拖拽修改 | 移动视角(V)：左键拖拽平移 | 正交路点：朝向限制为十字方向 | 固定路点(K)：点击路点固定/解锁 | 清空路点(C)：清空非 0 且未固定路点 | 绿色线：单向可达 | 蓝色线：双向可达 | 删除路点：点击路点 | 移动原点：点击或拖拽地图 | 鼠标滚轮：缩放 | 中键拖动/方向键：平移地图",
        "status": "模式：{mode}   工具：{tool}   缩放：{zoom:.2f}x   鼠标：{mouse}   选中：{selected}   位姿：{pose}",
        "tool_none": "普通",
        "tool_delete": "删除路点",
        "tool_origin": "移动原点",
        "tool_fix": "固定路点",
        "mouse_empty": "-",
        "selected_empty": "-",
        "saved": "已保存到 {path}",
        "save_error": "保存错误：{error}",
        "reloaded": "已重新加载路点",
        "undo_done": "已撤销",
        "redo_done": "已重做",
        "align_empty": "没有路点需要整理",
        "align_done": "智能整理完成",
        "delete_ref": "ID 0 基准点不能被删除",
        "fixed_ref": "固定路点不能被更改",
        "tool_rename": "修改序号",
        "rename_prompt": "修改序号工具已开启：请单击地图上的节点序号",
        "rename_title": "修改节点序号",
        "rename_label": "新的 ID：",
        "rename_conflict_title": "ID 冲突",
        "rename_conflict_message": "节点 ID {id} 已经存在，请重新输入。",
        "rename_invalid_title": "无效 ID",
        "rename_invalid_message": "节点 ID 必须是大于等于 0 的整数，ID 0 保留给原点，不能修改。",
        "rename_done": "已将节点 {old_id} 修改为 {new_id}",
        "delete_prompt": "删除工具已开启：请在地图上点击要删除的路点",
        "delete_done": "已删除路点 {id}",
        "edit_delete_done": "已删除路点 {id}；鼠标左键点击可重新设置该 ID",
        "reset_deleted": "重新设置路点 {id}: ({x:.2f}, {y:.2f}, {yaw:.2f})",
        "nudged": "已移动路点 {id}: ({x:.2f}, {y:.2f})",
        "nudge_missing": "请先选中路点，再使用方向键微调",
        "origin_nudge_missing": "请先选中 ID 0 原点，再使用方向键微调",
        "clear_empty": "没有可清空的非 0 路点",
        "clear_done": "已清空所有非 0 路点",
        "edit_ref": "ID 0 基准点不能被移动",
        "origin_prompt": "移动原点工具已开启：点击 ID 0 原点选中，方向键每次移动 0.01m",
        "origin_moved": "已移动原点：({x:.2f}, {y:.2f}, {yaw:.2f})",
        "selected_origin": "已选中原点，可使用方向键微调",
        "fix_prompt": "固定路点工具已开启：点击路点可固定或解锁",
        "fixed_node": "已固定路点 {id}",
        "unfixed_node": "已解锁路点 {id}",
        "view_prompt": "移动视角模式已开启：按住鼠标左键拖动画面",
        "selected_edit": "选中路点 {id} 进行编辑",
        "selected_pose": "位姿：({x:.2f}, {y:.2f}, {yaw:.2f})",
        "added": "添加路点 {id}: ({x:.2f}, {y:.2f}, {yaw:.2f})",
        "edited": "编辑路点 {id}: ({x:.2f}, {y:.2f}, {yaw:.2f})",
        "conn_start": "选择 {id} 作为连接起点",
        "conn_add": "连接：{src} -> {dst}",
        "conn_remove": "删除连接：{src} -> {dst}",
        "outside_add": "忽略地图外路点",
        "outside_edit": "忽略地图外编辑",
        "load_error": "加载路点错误：{error}",
        "map_load_error": "无法加载地图图像：{path}",
        "no_map": "请指定地图 YAML 文件",
    },
}


def initial_language():
    language = os.environ.get("ASR_LANGUAGE", "Chinese")
    return "English" if language == "English" else "Chinese"


def normalize_save_name(save_name):
    name = str(save_name or "").strip()
    if not name:
        name = DEFAULT_SAVE_NAME
    name = os.path.basename(name)
    if not name:
        name = DEFAULT_SAVE_NAME
    _, ext = os.path.splitext(name)
    if ext.lower() not in (".yaml", ".yml"):
        name = f"{name}.yaml"
    return name


class FileComboBox(QComboBox):
    def enterEvent(self, event):
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self.update()

    def showPopup(self):
        super().showPopup()
        self.update()

    def hidePopup(self):
        super().hidePopup()
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        color = QColor(194, 65, 12) if self.view().isVisible() or self.underMouse() else QColor(17, 17, 17)
        cx = self.width() - 18
        cy = self.height() / 2 + 1
        arrow = QPolygonF(
            [
                QPointF(cx - 6, cy - 3),
                QPointF(cx + 6, cy - 3),
                QPointF(cx, cy + 5),
            ]
        )
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(arrow)


class MapCanvas(QWidget):
    def __init__(self, editor):
        super().__init__()
        self.editor = editor
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(640, 420)

    def paintEvent(self, event):
        if self.width() <= 0 or self.height() <= 0:
            return
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(80, 80, 80))
        self.editor.paint_map(painter)

    def resizeEvent(self, event):
        if not self.editor.view_initialized:
            self.editor.fit_to_window()
            self.editor.view_initialized = True
        else:
            self.editor.clamp_offset()
        self.editor.update_status()
        super().resizeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        self.editor.schedule_restore_canvas()

    def mousePressEvent(self, event):
        self.setFocus()
        if event.button() == Qt.LeftButton:
            self.editor.begin_left_drag(event.x(), event.y())
        elif event.button() == Qt.RightButton:
            self.editor.right_click(event.x(), event.y())
        elif event.button() == Qt.MiddleButton:
            self.editor.begin_pan(event.x(), event.y())
        event.accept()

    def mouseMoveEvent(self, event):
        self.editor.mouse_pos = (event.x(), event.y())
        if self.editor.action == "pan":
            self.editor.update_pan(event.x(), event.y())
        elif self.editor.action in ("waypoint", "origin"):
            self.editor.update_waypoint_drag(event.x(), event.y())
        else:
            self.editor.update_status()
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.editor.finish_left_drag(event.x(), event.y())
        elif event.button() == Qt.MiddleButton:
            self.editor.finish_pan()
        event.accept()

    def mouseDoubleClickEvent(self, event):
        event.accept()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.editor.zoom_at(event.x(), event.y(), ZOOM_FACTOR)
        elif delta < 0:
            self.editor.zoom_at(event.x(), event.y(), 1.0 / ZOOM_FACTOR)
        self.editor.refresh()
        event.accept()


class WaypointEditor(QMainWindow):
    def __init__(self, map_yaml, save_name):
        super().__init__()
        self.language = initial_language()
        self.save_name_text = normalize_save_name(save_name)
        self.save_name = os.path.join(CONFIG_DIR, self.save_name_text)
        self.work_mode = "select"
        self.node_mode = "select"
        self.edit_mode = False
        self.deleted_edit_id = None
        self.suppress_file_selector_signal = False
        self.orthogonal_mode = False
        self.tool_mode = None
        self.edit_target = None
        self.edit_ready_id = None
        self.selected_id = None
        self.selected_from = None
        self.zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.mouse_pos = None
        self.action = None
        self.pan_last = None
        self.drag_start = None
        self.drag_current = None
        self.drag_start_view = None
        self.drag_current_view = None
        self.history = []
        self.redo_stack = []
        self.message = ""
        self.message_key = None
        self.message_kwargs = {}
        self.action_help_key = None
        self.view_initialized = False
        self.shortcuts = []

        self.load_map(map_yaml)
        self.waypoints = []
        self.load_existing_waypoints()
        if not self.waypoints:
            self.waypoints = [{"id": 0, "pose": [0.0, 0.0, 0.0], "to": []}]
        self.normalize_waypoints()

        self.canvas = MapCanvas(self)
        self.status_label = QLabel()
        self.status_label.setObjectName("statusLabel")
        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)
        self.action_help_label = QLabel()
        self.action_help_label.setObjectName("actionHelpLabel")
        self.action_help_label.setWordWrap(True)
        self.file_label = QLabel()
        self.file_selector = FileComboBox()
        self.file_selector.setEditable(True)
        self.file_selector.setInsertPolicy(QComboBox.NoInsert)
        self.file_selector.setMaxVisibleItems(12)
        self.file_selector.setFocusPolicy(Qt.StrongFocus)
        self.file_edit = self.file_selector.lineEdit()
        self.file_edit.setText(self.save_name_text)
        self.file_edit.setPlaceholderText(DEFAULT_SAVE_NAME)
        self.file_edit.setClearButtonEnabled(False)
        self.buttons = {}
        self.build_ui()
        self.setup_file_selector()
        self.install_shortcuts()
        self.apply_language()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        QTimer.singleShot(0, self.initial_fit)

    def text(self, key):
        return TEXT[self.language][key]

    def load_map(self, map_yaml):
        with open(map_yaml, "r") as f:
            map_data = yaml.safe_load(f)

        self.resolution = map_data["resolution"]
        self.origin = map_data["origin"]
        image_path = os.path.join(os.path.dirname(map_yaml), map_data["image"])

        self.map_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.map_raw is None:
            language = initial_language()
            print(TEXT[language]["map_load_error"].format(path=image_path))
            sys.exit(1)

        self.map_rgb = cv2.cvtColor(self.map_raw, cv2.COLOR_GRAY2RGB)
        self.refresh_map_image()

    def refresh_map_image(self):
        h, w = self.map_rgb.shape[:2]
        bytes_per_line = self.map_rgb.strides[0]
        self.map_image = QImage(
            self.map_rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888,
        ).copy()

    def build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)
        self.file_label.setObjectName("fileLabel")
        self.file_selector.setObjectName("fileSelector")
        self.file_edit.setObjectName("fileEdit")
        top_row.addWidget(self.file_label)
        top_row.addWidget(self.file_selector, 1)

        command_layout = QVBoxLayout()
        command_layout.setContentsMargins(0, 0, 0, 0)
        command_layout.setSpacing(8)
        primary_row = QHBoxLayout()
        primary_row.setSpacing(8)
        primary_row.setContentsMargins(0, 0, 0, 0)
        secondary_row = QHBoxLayout()
        secondary_row.setSpacing(8)
        secondary_row.setContentsMargins(0, 0, 0, 0)
        file_row = QHBoxLayout()
        file_row.setSpacing(8)
        file_row.setContentsMargins(0, 0, 0, 0)

        row_specs = [
            (
                primary_row,
                [
                    ("add_node", self.set_add_mode),
                    ("orthogonal_node", self.toggle_orthogonal_mode),
                    ("rename_node", self.toggle_rename_tool),
                    ("edit_node", self.set_edit_mode),
                    ("fix_node", self.toggle_fix_tool),
                    ("align", self.smart_align),
                    ("delete", self.toggle_delete_tool),
                    ("fit", self.fit_view),
                    ("move_view", self.set_view_mode),
                    ("move_origin", self.toggle_origin_tool),
                    ("clear_nodes", self.clear_non_reference_nodes),
                ],
            ),
            (
                secondary_row,
                [
                    ("undo", self.undo),
                    ("redo", self.redo),
                    ("zoom_in", self.zoom_in),
                    ("zoom_out", self.zoom_out),
                ],
            ),
            (
                file_row,
                [
                    ("reload", self.reload_waypoints),
                    ("quit", self.close),
                    ("save", self.save),
                    ("save_quit", self.save_and_quit),
                    ("language", self.toggle_language),
                ],
            ),
        ]

        for row, button_specs in row_specs:
            for name, callback in button_specs:
                button = self.create_button(name, callback)
                row.addWidget(button)
            row.addStretch(1)

        command_layout.addLayout(primary_row)
        command_layout.addWidget(self.action_help_label)
        command_layout.addLayout(secondary_row)
        command_layout.addLayout(file_row)

        hint_row = QHBoxLayout()
        hint_row.setContentsMargins(0, 0, 0, 0)
        hint_row.addWidget(self.hint_label, 1)

        self.status_label.setMinimumHeight(34)
        root.addLayout(top_row)
        root.addLayout(command_layout)
        root.addLayout(hint_row)
        root.addWidget(self.status_label)
        root.addWidget(self.canvas, 1)
        self.setCentralWidget(central)

        self.setStyleSheet(
            """
            QMainWindow {
                background: #f5f5f7;
            }
            QPushButton {
                padding: 7px 12px;
                border: 2px solid #111111;
                border-radius: 8px;
                background: #ffffff;
                color: #1d1d1f;
                font-size: 15px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #ffd6a3;
                border-color: #c2410c;
            }
            QPushButton:pressed {
                background: #fb923c;
                border-color: #9a3412;
            }
            QPushButton:checked {
                background: #0071e3;
                border-color: #111111;
                color: #ffffff;
            }
            QPushButton:checked:hover {
                background: #0a84ff;
                border-color: #c2410c;
            }
            QLabel {
                color: #1d1d1f;
                font-size: 15px;
            }
            QLineEdit {
                padding: 7px 10px;
                border: 1px solid #d2d2d7;
                border-radius: 8px;
                background: #ffffff;
                color: #1d1d1f;
                font-size: 15px;
                font-weight: 700;
            }
            QComboBox {
                padding: 7px 42px 7px 12px;
                border: 2px solid #111111;
                border-radius: 8px;
                background: #ffffff;
                color: #1d1d1f;
                font-size: 15px;
                font-weight: 700;
                min-width: 150px;
            }
            QComboBox QLineEdit {
                padding: 0;
                border: none;
                border-radius: 0;
                background: #ffffff;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 38px;
                border: none;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
                background: transparent;
            }
            QComboBox::drop-down:hover {
                background: #ffd6a3;
            }
            QComboBox::down-arrow {
                image: none;
                width: 0;
                height: 0;
            }
            QComboBox:hover {
                border-color: #c2410c;
            }
            QComboBox:on {
                border-color: #c2410c;
                background: #ffffff;
            }
            QComboBox QAbstractItemView {
                padding: 4px;
                border: 2px solid #111111;
                border-radius: 8px;
                background: #ffffff;
                color: #1d1d1f;
                selection-background-color: #ffd6a3;
                selection-color: #111111;
                outline: none;
            }
            QComboBox QAbstractItemView::item {
                min-height: 28px;
                padding: 6px 10px;
                border-radius: 6px;
            }
            QComboBox QAbstractItemView::item:hover {
                background: #ffd6a3;
                color: #111111;
            }
            QComboBox QAbstractItemView::item:selected {
                background: #fb923c;
                color: #111111;
            }
            #languageButton {
                padding: 0;
                min-width: 52px;
                max-width: 52px;
                min-height: 52px;
                max-height: 52px;
                border-radius: 12px;
                font-size: 15px;
                font-weight: 800;
            }
            #fileLabel {
                font-weight: 700;
            }
            #actionHelpLabel {
                padding: 0 2px 2px 2px;
                color: #3f3f46;
                font-size: 14px;
                font-weight: 700;
            }
            #statusLabel {
                padding: 6px 10px;
                background: #ffffff;
                border: 1px solid #e5e5ea;
                border-radius: 8px;
            }
            """
        )

    def create_button(self, name, callback):
        button = QPushButton()
        button.clicked.connect(callback)
        button.setMinimumHeight(34)
        button.setFocusPolicy(Qt.NoFocus)
        if name in (
            "add_node",
            "edit_node",
            "move_view",
            "delete",
            "move_origin",
            "fix_node",
            "rename_node",
            "orthogonal_node",
        ):
            button.setCheckable(True)
        if name == "language":
            button.setObjectName("languageButton")
            button.setFixedSize(52, 52)
        self.buttons[name] = button
        return button

    def install_shortcuts(self):
        shortcut_specs = [
            ("A", self.set_add_mode),
            ("E", self.set_edit_mode),
            ("V", self.set_view_mode),
            ("X", self.toggle_orthogonal_mode),
            ("O", self.toggle_origin_tool),
            ("Z", self.undo),
            ("Y", self.redo),
            ("S", self.save),
            ("L", self.reload_waypoints),
            ("G", self.smart_align),
            ("D", self.toggle_delete_tool),
            ("K", self.toggle_fix_tool),
            ("C", self.clear_non_reference_nodes),
            ("Q", self.save_and_quit),
            (Qt.Key_Escape, self.close),
            (Qt.Key_Plus, self.zoom_in),
            (Qt.Key_Equal, self.zoom_in),
            (Qt.Key_Minus, self.zoom_out),
            (Qt.Key_Underscore, self.zoom_out),
            ("F", self.fit_view),
            (Qt.Key_Left, lambda: self.arrow_key_move(-0.01, 0.0, PAN_STEP, 0)),
            (Qt.Key_Right, lambda: self.arrow_key_move(0.01, 0.0, -PAN_STEP, 0)),
            (Qt.Key_Up, lambda: self.arrow_key_move(0.0, 0.01, 0, PAN_STEP)),
            (Qt.Key_Down, lambda: self.arrow_key_move(0.0, -0.01, 0, -PAN_STEP)),
        ]

        for key, callback in shortcut_specs:
            sequence = QKeySequence(key)
            shortcut = QShortcut(sequence, self)
            shortcut.setContext(Qt.ApplicationShortcut)
            shortcut.activated.connect(lambda cb=callback: self.run_shortcut(cb))
            self.shortcuts.append(shortcut)

    def file_selector_has_focus(self):
        if not hasattr(self, "file_selector") or not hasattr(self, "file_edit"):
            return False
        focused = QApplication.focusWidget()
        if focused is not None:
            if focused is self.file_selector or focused is self.file_edit:
                return True
            if self.file_selector.isAncestorOf(focused) or self.file_edit.isAncestorOf(focused):
                return True
        return self.file_edit.hasFocus() or self.file_selector.hasFocus() or self.file_selector.view().isVisible()

    def run_shortcut(self, callback):
        if self.file_selector_has_focus():
            return
        callback()

    def initial_fit(self):
        if not self.view_initialized:
            self.fit_to_window()
            self.view_initialized = True
        self.refresh()

    def showEvent(self, event):
        super().showEvent(event)
        self.schedule_restore_canvas()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange and not self.isMinimized():
            self.schedule_restore_canvas()

    def schedule_restore_canvas(self):
        QTimer.singleShot(0, self.restore_canvas)
        QTimer.singleShot(80, self.restore_canvas)

    def restore_canvas(self):
        if not hasattr(self, "canvas") or self.canvas.width() <= 0 or self.canvas.height() <= 0:
            return
        self.refresh_map_image()
        self.clamp_offset()
        self.canvas.repaint()
        self.update_status()

    def apply_language(self):
        self.setWindowTitle(self.text("title"))
        for name in [
            "add_node",
            "orthogonal_node",
            "rename_node",
            "edit_node",
            "move_view",
            "move_origin",
            "fix_node",
            "undo",
            "redo",
            "save",
            "reload",
            "align",
            "delete",
            "clear_nodes",
            "zoom_in",
            "zoom_out",
            "fit",
            "save_quit",
            "quit",
            "language",
        ]:
            self.buttons[name].setText(self.text(name))
        self.file_label.setText(self.text("file_label"))
        self.file_edit.setPlaceholderText(self.text("file_placeholder"))
        self.refresh_file_selector()
        self.update_action_help()
        self.hint_label.setText(self.text("hint"))
        self.update_mode_buttons()
        self.update_tool_buttons()
        self.update_orthogonal_button()
        self.update_status()

    def toggle_language(self):
        self.language = "Chinese" if self.language == "English" else "English"
        self.refresh_message_language()
        self.apply_language()

    def setup_file_selector(self):
        self.file_selector.activated[int].connect(self.load_selected_network_file)
        self.file_edit.returnPressed.connect(self.reload_waypoints)
        self.refresh_file_selector()

    def available_network_files(self):
        if not os.path.isdir(CONFIG_DIR):
            return []

        network_files = []
        for filename in sorted(os.listdir(CONFIG_DIR)):
            if not filename.lower().endswith((".yaml", ".yml")):
                continue
            path = os.path.join(CONFIG_DIR, filename)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "r") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict) and isinstance(data.get("waypoints"), list):
                    network_files.append(filename)
            except Exception:
                continue
        return network_files

    def refresh_file_selector(self):
        if not hasattr(self, "file_selector"):
            return
        files = self.available_network_files()
        current = self.file_edit.text().strip() if hasattr(self, "file_edit") else self.save_name_text
        if not current:
            current = self.save_name_text
        self.suppress_file_selector_signal = True
        self.file_selector.clear()
        self.file_selector.addItems(files)
        if current in files:
            self.file_selector.setCurrentText(current)
        else:
            self.file_selector.setEditText(current)
        self.suppress_file_selector_signal = False

    def load_selected_network_file(self, index):
        if self.suppress_file_selector_signal or index < 0:
            return
        filename = self.file_selector.itemText(index).strip()
        if not filename:
            return
        self.file_edit.setText(normalize_save_name(filename))
        self.reload_waypoints()
        self.canvas.setFocus(Qt.MouseFocusReason)

    def set_message(self, message):
        self.message_key = None
        self.message_kwargs = {}
        self.message = str(message)
        print(self.message)
        self.update_status()

    def set_message_key(self, key, **kwargs):
        self.message_key = key
        self.message_kwargs = kwargs
        self.message = self.text(key).format(**kwargs)
        print(self.message)
        self.update_status()

    def clear_message_if_keys(self, *keys):
        if self.message_key in keys:
            self.message_key = None
            self.message_kwargs = {}
            self.message = ""
            self.update_status()

    def refresh_message_language(self):
        if self.message_key is not None:
            self.message = self.text(self.message_key).format(**self.message_kwargs)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress:
            self.restore_canvas_focus_from_file_edit(obj)
        return super().eventFilter(obj, event)

    def restore_canvas_focus_from_file_edit(self, obj):
        if not hasattr(self, "file_edit") or not self.file_edit.hasFocus():
            return
        if not isinstance(obj, QWidget):
            return
        if obj is self.file_selector or self.file_selector.isAncestorOf(obj):
            return
        if obj is self.file_edit or self.file_edit.isAncestorOf(obj):
            return
        if obj is self or self.isAncestorOf(obj):
            self.file_edit.clearFocus()
            if hasattr(self, "canvas"):
                self.canvas.setFocus(Qt.MouseFocusReason)

    def closeEvent(self, event):
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        super().closeEvent(event)

    def sync_save_name_from_editor(self):
        name = normalize_save_name(self.file_edit.text())
        self.save_name_text = name
        self.save_name = os.path.join(CONFIG_DIR, name)
        if self.file_edit.text() != name:
            self.file_edit.setText(name)
        return name

    def confirm_save(self):
        self.sync_save_name_from_editor()

        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Question)
        dialog.setWindowTitle(self.text("save_confirm_title"))
        message_key = "save_confirm_message" if os.path.exists(self.save_name) else "save_confirm_missing_message"
        dialog.setText(self.text(message_key).format(path=self.save_name))
        confirm_button = dialog.addButton(self.text("confirm"), QMessageBox.AcceptRole)
        dialog.addButton(self.text("cancel"), QMessageBox.RejectRole)
        dialog.setDefaultButton(confirm_button)
        dialog.setStyleSheet(self.dialog_style_sheet())
        dialog.exec_()
        return dialog.clickedButton() == confirm_button

    def dialog_style_sheet(self):
        return """
            QMessageBox {
                background: #f5f5f7;
            }
            QLabel {
                color: #1d1d1f;
                font-size: 15px;
                font-weight: 700;
            }
            QPushButton {
                min-width: 86px;
                min-height: 32px;
                padding: 7px 14px;
                border: 2px solid #111111;
                border-radius: 8px;
                background: #ffffff;
                color: #1d1d1f;
                font-size: 15px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #ffd6a3;
                border-color: #c2410c;
            }
            QPushButton:pressed {
                background: #fb923c;
                border-color: #9a3412;
            }
        """

    def refresh(self):
        self.clamp_offset()
        self.canvas.update()
        self.update_status()

    def update_status(self):
        if not hasattr(self, "status_label"):
            return

        mode = self.mode_text()
        tool = self.tool_text()
        selected = self.text("selected_empty") if self.selected_id is None else str(self.selected_id)
        pose_text = self.text("selected_empty")
        if self.selected_id is not None and self.is_waypoint_active(self.selected_id):
            pose = self.get_waypoint(self.selected_id)["pose"]
            pose_text = f"{pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}"
        mouse_text = self.text("mouse_empty")
        if self.mouse_pos is not None:
            mx, my = self.view_to_map(*self.mouse_pos)
            if self.is_inside_map(mx, my):
                wx, wy = self.pixel_to_world(mx, my)
                mouse_text = f"{wx:.2f}, {wy:.2f}"

        status = self.text("status").format(
            mode=mode,
            tool=tool,
            zoom=self.zoom,
            mouse=mouse_text,
            selected=selected,
            pose=pose_text,
        )
        if self.message:
            status = f"{status}   |   {self.message}"
        self.status_label.setText(status)

    def tool_text(self):
        if self.tool_mode == "delete":
            return self.text("tool_delete")
        if self.tool_mode == "origin":
            return self.text("tool_origin")
        if self.tool_mode == "fix":
            return self.text("tool_fix")
        if self.tool_mode == "rename":
            return self.text("tool_rename")
        return self.text("tool_none")

    def mode_text(self):
        if self.work_mode == "edit":
            return self.text("mode_edit")
        if self.work_mode == "pan":
            return self.text("mode_pan")
        if self.work_mode == "add":
            return self.text("mode_add")
        return self.text("mode_select")

    def set_tool_mode(self, mode):
        self.tool_mode = mode
        self.edit_target = None
        self.selected_from = None
        self.drag_start = None
        self.drag_current = None
        self.drag_start_view = None
        self.drag_current_view = None
        self.action = None
        self.update_tool_buttons()

        if mode == "delete":
            self.set_message_key("delete_prompt")
        elif mode == "origin":
            self.set_message_key("origin_prompt")
        elif mode == "fix":
            self.action_help_key = "fix_node_help"
            self.update_action_help()
            self.set_message_key("fix_prompt")
        elif mode == "rename":
            self.action_help_key = "rename_node_help"
            self.update_action_help()
            self.set_message_key("rename_prompt")
        else:
            if self.action_help_key in ("fix_node_help", "rename_node_help"):
                self.action_help_key = None
                self.update_action_help()
            self.clear_message_if_keys("delete_prompt", "origin_prompt", "fix_prompt", "rename_prompt")
            self.update_status()
            self.refresh()

    def update_tool_buttons(self):
        if not hasattr(self, "buttons"):
            return
        if "delete" in self.buttons:
            self.buttons["delete"].setChecked(self.tool_mode == "delete")
        if "move_origin" in self.buttons:
            self.buttons["move_origin"].setChecked(self.tool_mode == "origin")
        if "fix_node" in self.buttons:
            self.buttons["fix_node"].setChecked(self.tool_mode == "fix")
        if "rename_node" in self.buttons:
            self.buttons["rename_node"].setChecked(self.tool_mode == "rename")

    def update_mode_buttons(self):
        if not hasattr(self, "buttons"):
            return
        if "add_node" in self.buttons:
            self.buttons["add_node"].setChecked(self.work_mode == "add")
        if "edit_node" in self.buttons:
            self.buttons["edit_node"].setChecked(self.work_mode == "edit")
        if "move_view" in self.buttons:
            self.buttons["move_view"].setChecked(self.work_mode == "pan")

    def update_orthogonal_button(self):
        if hasattr(self, "buttons") and "orthogonal_node" in self.buttons:
            self.buttons["orthogonal_node"].setChecked(self.orthogonal_mode)

    def toggle_orthogonal_mode(self):
        self.orthogonal_mode = not self.orthogonal_mode
        self.update_orthogonal_button()
        self.update_action_help()
        self.refresh()

    def update_action_help(self):
        if not hasattr(self, "action_help_label"):
            return
        if self.action_help_key is not None:
            help_text = self.text(self.action_help_key)
        elif self.work_mode == "edit":
            help_text = self.text("edit_node_help")
        elif self.work_mode == "pan":
            help_text = self.text("move_view_help")
        elif self.work_mode == "add":
            help_text = self.text("add_node_help")
        else:
            help_text = ""

        if self.orthogonal_mode:
            orthogonal_help = self.text("orthogonal_help")
            help_text = f"{help_text} | {orthogonal_help}" if help_text else orthogonal_help
        self.action_help_label.setText(help_text)

    def snap_yaw(self, yaw):
        if yaw is None or not self.orthogonal_mode:
            return yaw
        directions = [0.0, math.pi / 2, math.pi, -math.pi / 2]
        return round(min(directions, key=lambda angle: abs(math.atan2(math.sin(yaw - angle), math.cos(yaw - angle)))), 2)

    def toggle_delete_tool(self):
        if self.edit_mode:
            if self.selected_id == 0:
                self.set_message_key("delete_ref")
                return
            if self.selected_id is not None and self.is_waypoint_active(self.selected_id):
                self.delete_waypoint(self.selected_id)
                return
        self.set_tool_mode(None if self.tool_mode == "delete" else "delete")

    def toggle_origin_tool(self):
        self.set_tool_mode(None if self.tool_mode == "origin" else "origin")

    def toggle_fix_tool(self):
        self.set_tool_mode(None if self.tool_mode == "fix" else "fix")

    def toggle_rename_tool(self):
        self.set_tool_mode(None if self.tool_mode == "rename" else "rename")

    def clear_edit_ready(self):
        self.edit_ready_id = None

    def compact_deleted_waypoints(self):
        deleted_ids = [wp["id"] for wp in self.waypoints if wp.get("_deleted", False)]
        if not deleted_ids:
            return
        deleted = set(deleted_ids)
        old_to_new = {}
        compacted = []
        for wp in self.waypoints:
            if wp.get("_deleted", False):
                continue
            old_to_new[wp["id"]] = len(compacted)
            compacted.append(wp)

        for new_id, wp in enumerate(compacted):
            wp["id"] = new_id
            wp["_deleted"] = False
            wp.setdefault("_fixed", False)
            wp["to"] = [old_to_new[t] for t in wp["to"] if t not in deleted and t in old_to_new]

        self.waypoints = compacted or [{"id": 0, "pose": [0.0, 0.0, 0.0], "to": [], "_deleted": False, "_fixed": False}]
        self.selected_id = None
        self.selected_from = None
        self.edit_target = None
        self.clear_edit_ready()
        self.deleted_edit_id = None

    def set_add_mode(self):
        if self.edit_mode:
            self.compact_deleted_waypoints()
        if self.work_mode == "add":
            self.set_select_mode()
            return
        self.node_mode = "add"
        self.work_mode = "add"
        self.edit_mode = False
        self.edit_target = None
        self.clear_edit_ready()
        self.set_tool_mode(None)
        self.clear_message_if_keys("view_prompt")
        self.update_mode_buttons()
        self.update_action_help()
        self.refresh()

    def set_select_mode(self):
        if self.edit_mode:
            self.compact_deleted_waypoints()
        self.node_mode = "select"
        self.work_mode = "select"
        self.edit_mode = False
        self.edit_target = None
        self.clear_edit_ready()
        self.set_tool_mode(None)
        self.clear_message_if_keys("view_prompt")
        self.update_mode_buttons()
        self.update_action_help()
        self.refresh()

    def set_edit_mode(self):
        if self.work_mode == "edit":
            self.set_select_mode()
            return
        self.node_mode = "edit"
        self.work_mode = "edit"
        self.edit_target = None
        if not self.edit_mode:
            self.deleted_edit_id = None
            self.clear_edit_ready()
        self.edit_mode = True
        self.set_tool_mode(None)
        self.clear_message_if_keys("view_prompt")
        self.update_mode_buttons()
        self.update_action_help()
        self.refresh()

    def set_view_mode(self):
        if self.work_mode == "pan":
            if self.node_mode == "edit":
                self.set_edit_mode()
            elif self.node_mode == "add":
                self.set_add_mode()
            else:
                self.set_select_mode()
            return

        if self.work_mode in ("add", "edit", "select"):
            self.node_mode = self.work_mode
        if self.edit_mode:
            self.compact_deleted_waypoints()
        self.work_mode = "pan"
        self.edit_mode = False
        self.edit_target = None
        self.clear_edit_ready()
        self.set_tool_mode(None)
        self.set_message_key("view_prompt")
        self.update_mode_buttons()
        self.update_action_help()
        self.refresh()

    def normalize_waypoints(self):
        used_ids = set()
        for i, wp in enumerate(self.waypoints):
            waypoint_id = int(wp.get("id", i))
            if waypoint_id in used_ids:
                waypoint_id = i
            wp["id"] = waypoint_id
            used_ids.add(waypoint_id)
            wp.setdefault("_deleted", False)
            wp.setdefault("_fixed", False)
            wp.setdefault("pose", [0.0, 0.0, 0.0])
            while len(wp["pose"]) < 3:
                wp["pose"].append(0.0)
            wp.setdefault("to", [])
            wp["to"] = [int(t) for t in wp["to"] if isinstance(t, int) or str(t).isdigit()]

    def get_waypoint(self, waypoint_id):
        for wp in self.waypoints:
            if wp["id"] == waypoint_id:
                return wp
        return None

    def is_waypoint_active(self, waypoint_id):
        wp = self.get_waypoint(waypoint_id)
        return wp is not None and not wp.get("_deleted", False)

    def active_waypoint_count(self):
        return sum(1 for wp in self.waypoints if not wp.get("_deleted", False))

    def next_available_id(self):
        used = {wp["id"] for wp in self.waypoints if not wp.get("_deleted", False)}
        waypoint_id = 0
        while waypoint_id in used:
            waypoint_id += 1
        return waypoint_id

    def clearable_waypoint_count(self):
        return sum(
            1
            for wp in self.waypoints
            if wp["id"] != 0 and not wp.get("_deleted", False) and not wp.get("_fixed", False)
        )

    def load_existing_waypoints(self):
        if os.path.exists(self.save_name):
            try:
                with open(self.save_name, "r") as f:
                    data = yaml.safe_load(f)
                    if data and "waypoints" in data:
                        self.waypoints = data["waypoints"]
            except Exception as e:
                self.set_message_key("load_error", error=e)

    def reload_waypoints(self):
        self.sync_save_name_from_editor()
        self.refresh_file_selector()
        self.waypoints = []
        self.load_existing_waypoints()
        if not self.waypoints:
            self.waypoints = [{"id": 0, "pose": [0.0, 0.0, 0.0], "to": []}]
        self.normalize_waypoints()
        self.selected_id = None
        self.selected_from = None
        self.edit_target = None
        self.clear_edit_ready()
        self.deleted_edit_id = None
        self.set_tool_mode(None)
        self.set_message_key("reloaded")
        self.refresh()

    def push_history(self):
        self.history.append(copy.deepcopy(self.waypoints))
        self.redo_stack.clear()

    def undo(self):
        if self.history:
            self.redo_stack.append(copy.deepcopy(self.waypoints))
            self.waypoints = self.history.pop()
            self.normalize_waypoints()
            self.selected_id = None
            self.selected_from = None
            self.edit_target = None
            self.clear_edit_ready()
            self.deleted_edit_id = None
            self.set_message_key("undo_done")
            self.refresh()

    def redo(self):
        if self.redo_stack:
            self.history.append(copy.deepcopy(self.waypoints))
            self.waypoints = self.redo_stack.pop()
            self.normalize_waypoints()
            self.selected_id = None
            self.selected_from = None
            self.edit_target = None
            self.clear_edit_ready()
            self.deleted_edit_id = None
            self.set_message_key("redo_done")
            self.refresh()

    def clear_non_reference_nodes(self):
        if self.clearable_waypoint_count() <= 0:
            self.set_message_key("clear_empty")
            return

        self.push_history()
        keep_ids = {0}
        keep_ids.update(wp["id"] for wp in self.waypoints if wp.get("_fixed", False) and not wp.get("_deleted", False))
        old_to_new = {}
        kept = []
        for wp in self.waypoints:
            if wp["id"] in keep_ids:
                old_to_new[wp["id"]] = len(kept)
                kept.append(copy.deepcopy(wp))
        for new_id, wp in enumerate(kept):
            wp["id"] = new_id
            wp["_deleted"] = False
            wp["to"] = [old_to_new[t] for t in wp["to"] if t in old_to_new]
        self.waypoints = kept
        self.selected_id = None
        self.selected_from = None
        self.edit_target = None
        self.clear_edit_ready()
        self.deleted_edit_id = None
        self.set_tool_mode(None)
        self.set_message_key("clear_done")
        self.refresh()

    def delete_selected(self):
        if self.selected_id is None:
            return
        self.delete_waypoint(self.selected_id)

    def delete_waypoint(self, waypoint_id):
        if waypoint_id is None:
            return
        if waypoint_id == 0:
            self.set_message_key("delete_ref")
            return
        wp = self.get_waypoint(waypoint_id)
        if wp is None:
            return
        if wp.get("_fixed", False):
            self.set_message_key("fixed_ref")
            return
        if not self.is_waypoint_active(waypoint_id):
            return

        if self.edit_mode:
            self.delete_waypoint_for_reset(waypoint_id)
            return

        self.push_history()
        deleted_id = waypoint_id
        old_ids = [wp["id"] for wp in self.waypoints if wp["id"] != deleted_id]
        id_map = {old_id: new_id for new_id, old_id in enumerate(old_ids)}
        self.waypoints = [wp for wp in self.waypoints if wp["id"] != deleted_id]
        for wp in self.waypoints:
            old_id = wp["id"]
            wp["id"] = id_map[old_id]
            wp["to"] = [id_map[t] for t in wp["to"] if t != deleted_id and t in id_map]

        self.selected_id = None
        self.selected_from = None
        self.clear_edit_ready()
        self.set_message_key("delete_done", id=deleted_id)
        self.refresh()

    def delete_waypoint_for_reset(self, waypoint_id):
        if self.deleted_edit_id is not None:
            self.set_message_key("edit_delete_done", id=self.deleted_edit_id)
            return

        self.push_history()
        wp = self.get_waypoint(waypoint_id)
        if wp is None:
            return
        wp["_deleted"] = True
        wp["to"] = []
        for other in self.waypoints:
            other["to"] = [t for t in other["to"] if t != waypoint_id]
        self.deleted_edit_id = waypoint_id
        self.selected_id = None
        self.selected_from = None
        self.edit_target = None
        self.clear_edit_ready()
        self.set_tool_mode(None)
        self.set_message_key("edit_delete_done", id=waypoint_id)
        self.refresh()

    def pixel_to_world(self, px, py):
        h, _ = self.map_raw.shape[:2]
        x = px * self.resolution + self.origin[0]
        y = (h - py) * self.resolution + self.origin[1]
        return round(x, 2), round(y, 2)

    def world_to_pixel(self, x, y):
        h, _ = self.map_raw.shape[:2]
        px = (x - self.origin[0]) / self.resolution
        py = h - (y - self.origin[1]) / self.resolution
        return px, py

    def canvas_size(self):
        if hasattr(self, "canvas"):
            return max(1, self.canvas.width()), max(1, self.canvas.height())
        return WINDOW_WIDTH, WINDOW_HEIGHT

    def map_to_view(self, px, py):
        vx = px * self.zoom + self.offset_x
        vy = py * self.zoom + self.offset_y
        return vx, vy

    def view_to_map(self, vx, vy):
        px = (vx - self.offset_x) / self.zoom
        py = (vy - self.offset_y) / self.zoom
        return px, py

    def clamp_offset(self):
        canvas_w, canvas_h = self.canvas_size()
        h, w = self.map_raw.shape[:2]
        scaled_w = w * self.zoom
        scaled_h = h * self.zoom

        if scaled_w <= canvas_w:
            self.offset_x = (canvas_w - scaled_w) / 2.0
        else:
            self.offset_x = min(0.0, max(canvas_w - scaled_w, self.offset_x))

        if scaled_h <= canvas_h:
            self.offset_y = (canvas_h - scaled_h) / 2.0
        else:
            self.offset_y = min(0.0, max(canvas_h - scaled_h, self.offset_y))

    def fit_to_window(self):
        canvas_w, canvas_h = self.canvas_size()
        h, w = self.map_raw.shape[:2]
        margin = 30
        scale_x = max(1, canvas_w - margin) / w
        scale_y = max(1, canvas_h - margin) / h
        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, min(scale_x, scale_y)))
        self.offset_x = (canvas_w - w * self.zoom) / 2.0
        self.offset_y = (canvas_h - h * self.zoom) / 2.0

    def fit_view(self):
        self.fit_to_window()
        self.refresh()

    def zoom_anchor(self):
        canvas_w, canvas_h = self.canvas_size()
        if self.mouse_pos is not None:
            x, y = self.mouse_pos
            if 0 <= x <= canvas_w and 0 <= y <= canvas_h:
                return x, y
        return canvas_w / 2.0, canvas_h / 2.0

    def zoom_at(self, vx, vy, factor):
        old_zoom = self.zoom
        new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-6:
            return

        map_x, map_y = self.view_to_map(vx, vy)
        self.zoom = new_zoom
        self.offset_x = vx - map_x * self.zoom
        self.offset_y = vy - map_y * self.zoom
        self.clamp_offset()

    def zoom_in(self):
        x, y = self.zoom_anchor()
        self.zoom_at(x, y, ZOOM_FACTOR)
        self.refresh()

    def zoom_out(self):
        x, y = self.zoom_anchor()
        self.zoom_at(x, y, 1.0 / ZOOM_FACTOR)
        self.refresh()

    def pan(self, dx, dy):
        self.offset_x += dx
        self.offset_y += dy
        self.refresh()

    def arrow_key_move(self, world_dx, world_dy, pan_dx, pan_dy):
        if self.tool_mode == "origin":
            if self.selected_id != 0 or not self.is_waypoint_active(0):
                self.set_message_key("origin_nudge_missing")
                return
            self.push_history()
            wp = self.get_waypoint(0)
            if wp is None:
                return
            wp["pose"][0] = round(wp["pose"][0] + world_dx, 2)
            wp["pose"][1] = round(wp["pose"][1] + world_dy, 2)
            self.set_message_key("origin_moved", x=wp["pose"][0], y=wp["pose"][1], yaw=wp["pose"][2])
            self.refresh()
            return

        if self.edit_mode and self.tool_mode is None:
            selected_wp = self.get_waypoint(self.selected_id)
            if selected_wp is not None and selected_wp.get("_fixed", False):
                self.set_message_key("fixed_ref")
                return
            if self.selected_id is None or self.selected_id == 0 or not self.is_waypoint_active(self.selected_id):
                self.set_message_key("nudge_missing")
                return
            self.push_history()
            wp = self.get_waypoint(self.selected_id)
            if wp is None:
                return
            wp["pose"][0] = round(wp["pose"][0] + world_dx, 2)
            wp["pose"][1] = round(wp["pose"][1] + world_dy, 2)
            self.set_message_key("nudged", id=self.selected_id, x=wp["pose"][0], y=wp["pose"][1])
            self.refresh()
            return

        self.pan(pan_dx, pan_dy)

    def is_inside_map(self, px, py):
        h, w = self.map_raw.shape[:2]
        return 0 <= px < w and 0 <= py < h

    def find_nearest_view(self, vx, vy, thresh=12, include_reference=True):
        nearest_id = None
        nearest_dist = thresh
        for wp in self.waypoints:
            if wp.get("_deleted", False):
                continue
            if not include_reference and wp["id"] == 0:
                continue
            px, py = self.world_to_pixel(wp["pose"][0], wp["pose"][1])
            sx, sy = self.map_to_view(px, py)
            dist = math.hypot(sx - vx, sy - vy)
            if dist <= nearest_dist:
                nearest_dist = dist
                nearest_id = wp["id"]
        return nearest_id

    def show_info_dialog(self, title_key, message_key, **kwargs):
        QMessageBox.warning(
            self,
            self.text(title_key),
            self.text(message_key).format(**kwargs),
        )

    def rename_click(self, vx, vy):
        if self.tool_mode != "rename":
            return
        idx = self.find_nearest_view(vx, vy, thresh=28)
        if idx is None or not self.is_waypoint_active(idx):
            return
        self.rename_waypoint_id(idx)

    def double_click(self, vx, vy):
        self.rename_click(vx, vy)

    def rename_waypoint_id(self, old_id):
        if old_id == 0:
            self.show_info_dialog("rename_invalid_title", "rename_invalid_message")
            return

        current_value = old_id
        while True:
            new_id, ok = QInputDialog.getInt(
                self,
                self.text("rename_title"),
                self.text("rename_label"),
                current_value,
                0,
                9999,
                1,
            )
            if not ok:
                return
            if new_id == old_id:
                return

            current_value = new_id
            if new_id < 0 or (new_id == 0 and old_id != 0):
                self.show_info_dialog("rename_invalid_title", "rename_invalid_message")
                continue

            conflict = False
            for wp in sorted(self.waypoints, key=lambda item: item["id"]):
                if wp.get("_deleted", False):
                    continue
                if wp["id"] == new_id:
                    conflict = True
                    break
            if conflict:
                self.show_info_dialog("rename_conflict_title", "rename_conflict_message", id=new_id)
                continue
            break

        self.push_history()
        id_map = {}
        for wp in self.waypoints:
            current_id = wp["id"]
            id_map[current_id] = new_id if current_id == old_id else current_id
            if current_id == old_id:
                wp["id"] = new_id

        for wp in self.waypoints:
            wp["to"] = [id_map.get(t, t) for t in wp["to"]]

        self.waypoints.sort(key=lambda wp: wp["id"])
        self.selected_id = new_id
        self.selected_from = None if self.selected_from == old_id else id_map.get(self.selected_from, self.selected_from)
        self.edit_target = None if self.edit_target == old_id else id_map.get(self.edit_target, self.edit_target)
        self.edit_ready_id = None if self.edit_ready_id == old_id else id_map.get(self.edit_ready_id, self.edit_ready_id)
        self.deleted_edit_id = None if self.deleted_edit_id == old_id else id_map.get(self.deleted_edit_id, self.deleted_edit_id)
        self.set_message_key("rename_done", old_id=old_id, new_id=new_id)
        self.refresh()

    def smart_align(self):
        self.action_help_key = "align_help"
        self.update_action_help()
        if self.active_waypoint_count() < 2:
            self.set_message_key("align_empty")
            return

        self.push_history()
        origin_wp = self.get_waypoint(0)
        if origin_wp is None:
            return
        origin_pose = copy.deepcopy(origin_wp["pose"])
        fixed_poses = {
            wp["id"]: copy.deepcopy(wp["pose"])
            for wp in self.waypoints
            if wp.get("_fixed", False) and not wp.get("_deleted", False)
        }
        for wp in self.waypoints:
            if wp.get("_deleted", False):
                continue
            px, py = wp["pose"][0], wp["pose"][1]
            for target_id in wp["to"]:
                if target_id == 0:
                    continue
                target = self.get_waypoint(target_id)
                if target is not None and not target.get("_fixed", False):
                    tx, ty = target["pose"][0], target["pose"][1]
                    dx = tx - px
                    dy = ty - py
                    dist = math.hypot(dx, dy)
                    if dist > 0:
                        if abs(dx) > abs(dy):
                            target["pose"][1] = round(py, 2)
                        else:
                            target["pose"][0] = round(px, 2)

        origin_wp = self.get_waypoint(0)
        if origin_wp is not None:
            origin_wp["pose"] = origin_pose
        for waypoint_id, pose in fixed_poses.items():
            fixed_wp = self.get_waypoint(waypoint_id)
            if fixed_wp is not None and not fixed_wp.get("_deleted", False):
                fixed_wp["pose"] = pose
        self.set_message_key("align_done")
        self.refresh()

    def toggle_edit_mode(self):
        if self.work_mode == "edit":
            self.set_add_mode()
        else:
            self.set_edit_mode()

    def begin_pan(self, vx, vy):
        self.action = "pan"
        self.pan_last = (vx, vy)

    def update_pan(self, vx, vy):
        if self.pan_last is None:
            return
        dx = vx - self.pan_last[0]
        dy = vy - self.pan_last[1]
        self.pan_last = (vx, vy)
        self.offset_x += dx
        self.offset_y += dy
        self.refresh()

    def finish_pan(self):
        self.action = None
        self.pan_last = None

    def begin_left_drag(self, vx, vy):
        self.mouse_pos = (vx, vy)

        if self.tool_mode == "rename":
            self.cancel_current_drag()
            self.pan_last = None
            self.rename_click(vx, vy)
            return

        if self.work_mode == "pan" and self.tool_mode is None:
            self.begin_pan(vx, vy)
            return

        mx, my = self.view_to_map(vx, vy)

        if self.tool_mode == "delete":
            idx = self.find_nearest_view(vx, vy, include_reference=False)
            if idx is not None:
                self.selected_id = idx
                self.delete_waypoint(idx)
            else:
                reference_id = self.find_nearest_view(vx, vy)
                if reference_id == 0:
                    self.set_message_key("delete_ref")
                else:
                    self.refresh()
            return

        if self.tool_mode == "fix":
            idx = self.find_nearest_view(vx, vy)
            if idx is not None and self.is_waypoint_active(idx):
                self.toggle_waypoint_fixed(idx)
            else:
                self.refresh()
            return

        if self.tool_mode == "origin":
            idx = self.find_nearest_view(vx, vy)
            if idx != 0:
                self.edit_target = None
                self.clear_edit_ready()
                self.action = None
                self.set_message_key("origin_nudge_missing")
                self.refresh()
                return
            self.selected_id = 0
            if self.edit_ready_id != 0:
                self.edit_ready_id = 0
                self.edit_target = None
                self.set_message_key("selected_origin")
                self.refresh()
                return
            if not self.is_inside_map(mx, my):
                self.edit_target = None
                self.action = None
                self.refresh()
                return
            self.action = "origin"
            self.drag_start = (mx, my)
            self.drag_current = (mx, my)
            self.drag_start_view = (vx, vy)
            self.drag_current_view = (vx, vy)
            self.edit_target = 0
            self.set_message_key("selected_origin")
            self.refresh()
            return

        if not self.is_inside_map(mx, my):
            self.selected_id = None
            self.edit_target = None
            self.clear_edit_ready()
            self.action = None
            self.refresh()
            return

        self.action = "waypoint"
        self.drag_start = (mx, my)
        self.drag_current = (mx, my)
        self.drag_start_view = (vx, vy)
        self.drag_current_view = (vx, vy)

        nearest = self.find_nearest_view(vx, vy)
        if self.work_mode == "select":
            if nearest is not None and self.is_waypoint_active(nearest):
                self.selected_id = nearest
                self.edit_target = None
                self.clear_edit_ready()
                self.print_selected_pose(nearest)
            else:
                self.selected_id = None
                self.edit_target = None
                self.clear_edit_ready()
                self.refresh()
            self.cancel_current_drag()
            return

        if nearest is not None:
            if self.edit_mode:
                if nearest == 0:
                    self.selected_id = nearest
                    self.clear_edit_ready()
                    self.set_message_key("edit_ref")
                    self.edit_target = None
                    self.cancel_current_drag()
                    return
                nearest_wp = self.get_waypoint(nearest)
                if nearest_wp is not None and nearest_wp.get("_fixed", False):
                    self.selected_id = nearest
                    self.clear_edit_ready()
                    self.set_message_key("fixed_ref")
                    self.edit_target = None
                    self.cancel_current_drag()
                elif self.edit_ready_id != nearest:
                    self.selected_id = nearest
                    self.edit_ready_id = nearest
                    self.edit_target = None
                    self.set_message_key("selected_edit", id=nearest)
                    self.cancel_current_drag()
                else:
                    self.selected_id = nearest
                    self.edit_target = nearest
                    self.set_message_key("selected_edit", id=nearest)
            else:
                self.selected_id = nearest
        else:
            self.selected_id = None
            self.edit_target = None
            self.clear_edit_ready()
        self.refresh()

    def print_selected_pose(self, waypoint_id):
        if not self.is_waypoint_active(waypoint_id):
            return
        wp = self.get_waypoint(waypoint_id)
        if wp is None:
            return
        pose = wp["pose"]
        print(self.text("selected_pose").format(x=pose[0], y=pose[1], yaw=pose[2]))
        self.update_status()

    def cancel_current_drag(self):
        self.drag_start = None
        self.drag_current = None
        self.drag_start_view = None
        self.drag_current_view = None
        self.action = None

    def toggle_waypoint_fixed(self, waypoint_id):
        if not self.is_waypoint_active(waypoint_id):
            return
        self.push_history()
        wp = self.get_waypoint(waypoint_id)
        if wp is None:
            return
        wp["_fixed"] = not wp.get("_fixed", False)
        self.selected_id = waypoint_id
        if wp["_fixed"]:
            self.set_message_key("fixed_node", id=waypoint_id)
        else:
            self.set_message_key("unfixed_node", id=waypoint_id)
        self.refresh()

    def update_waypoint_drag(self, vx, vy):
        self.mouse_pos = (vx, vy)
        if self.drag_start is None:
            return
        self.drag_current = self.view_to_map(vx, vy)
        self.drag_current_view = (vx, vy)
        self.refresh()

    def finish_left_drag(self, vx, vy):
        self.mouse_pos = (vx, vy)
        if self.action == "pan":
            self.finish_pan()
            return

        if self.action not in ("waypoint", "origin") or self.drag_start is None:
            return

        end_mx, end_my = self.view_to_map(vx, vy)
        dx = end_mx - self.drag_start[0]
        dy = end_my - self.drag_start[1]
        view_dx = vx - self.drag_start_view[0]
        view_dy = vy - self.drag_start_view[1]
        dist = math.hypot(view_dx, view_dy)
        wx, wy = self.pixel_to_world(*self.drag_start)
        yaw = round(math.atan2(-dy, dx), 2) if dist > DRAG_THRESHOLD else None
        yaw = self.snap_yaw(yaw)

        if self.edit_mode and self.deleted_edit_id is not None:
            if not self.is_inside_map(*self.drag_start):
                self.set_message_key("outside_edit")
            else:
                self.reset_deleted_waypoint(self.deleted_edit_id, wx, wy, yaw)
        elif self.action == "origin":
            self.move_origin(wx, wy, yaw)
        elif self.edit_mode and self.edit_target is not None:
            if self.edit_target == 0:
                self.set_message_key("edit_ref")
            elif not self.is_inside_map(*self.drag_start):
                self.set_message_key("outside_edit")
            else:
                self.push_history()
                wp = self.get_waypoint(self.edit_target)
                if wp is None:
                    return
                wp["pose"][0] = wx
                wp["pose"][1] = wy
                if yaw is not None:
                    wp["pose"][2] = yaw
                self.clear_edit_ready()
                self.set_message_key(
                    "edited",
                    id=self.edit_target,
                    x=wx,
                    y=wy,
                    yaw=wp["pose"][2],
                )
        elif self.work_mode == "add":
            if not self.is_inside_map(*self.drag_start):
                self.set_message_key("outside_add")
            else:
                self.compact_deleted_waypoints()
                self.push_history()
                new_id = self.next_available_id()
                yaw_value = yaw if yaw is not None else 0.0
                self.waypoints.append(
                    {
                        "id": new_id,
                        "pose": [wx, wy, round(yaw_value, 2)],
                        "to": [],
                    }
                )
                self.selected_id = new_id
                self.set_message_key(
                    "added",
                    id=new_id,
                    x=wx,
                    y=wy,
                    yaw=yaw_value,
                )

        self.drag_start = None
        self.drag_current = None
        self.drag_start_view = None
        self.drag_current_view = None
        self.edit_target = None
        self.action = None
        self.refresh()

    def reset_deleted_waypoint(self, waypoint_id, wx, wy, yaw):
        self.push_history()
        wp = self.get_waypoint(waypoint_id)
        if wp is None:
            return
        wp["id"] = waypoint_id
        wp["_deleted"] = False
        wp["pose"] = [wx, wy, round(yaw if yaw is not None else 0.0, 2)]
        wp["to"] = []
        self.deleted_edit_id = None
        self.selected_id = waypoint_id
        self.edit_target = None
        self.set_message_key("reset_deleted", id=waypoint_id, x=wx, y=wy, yaw=wp["pose"][2])

    def move_origin(self, wx, wy, yaw):
        if not self.waypoints:
            return

        self.push_history()
        wp = self.get_waypoint(0)
        if wp is None:
            return
        wp["pose"][0] = wx
        wp["pose"][1] = wy
        if yaw is not None:
            wp["pose"][2] = yaw
        self.selected_id = 0
        self.clear_edit_ready()
        self.edit_target = None
        self.set_message_key("origin_moved", x=wx, y=wy, yaw=wp["pose"][2])

    def right_click(self, vx, vy):
        if self.tool_mode is not None:
            return

        self.mouse_pos = (vx, vy)
        idx = self.find_nearest_view(vx, vy)
        if idx is None or not self.is_waypoint_active(idx):
            return

        self.selected_id = idx
        if self.selected_from is None:
            self.selected_from = idx
            self.set_message_key("conn_start", id=idx)
        else:
            if idx != self.selected_from:
                if not self.is_waypoint_active(self.selected_from):
                    self.selected_from = None
                    self.refresh()
                    return
                self.push_history()
                source_wp = self.get_waypoint(self.selected_from)
                if source_wp is None:
                    self.selected_from = None
                    self.refresh()
                    return
                if idx not in source_wp["to"]:
                    source_wp["to"].append(idx)
                    self.set_message_key("conn_add", src=self.selected_from, dst=idx)
                else:
                    source_wp["to"].remove(idx)
                    self.set_message_key("conn_remove", src=self.selected_from, dst=idx)
            self.selected_from = None
        self.refresh()

    def save(self):
        try:
            self.sync_save_name_from_editor()
            self.compact_deleted_waypoints()
            existed = os.path.exists(self.save_name)
            if not self.confirm_save():
                return False
            self.write_waypoints_file()
            self.refresh_file_selector()
            message_key = "saved" if existed else "created_file"
            self.set_message_key(message_key, path=self.save_name)
            return True
        except Exception as e:
            self.set_message_key("save_error", error=e)
            return False

    def write_waypoints_file(self):
        os.makedirs(os.path.dirname(self.save_name), exist_ok=True)
        data_to_save = {"waypoints": []}
        for wp in self.waypoints:
            if wp.get("_deleted", False):
                continue
            pose = [
                round(wp["pose"][0], 2),
                round(wp["pose"][1], 2),
                round(wp["pose"][2], 2),
            ]
            data_to_save["waypoints"].append(
                {
                    "id": int(wp["id"]),
                    "pose": pose,
                    "to": [int(t) for t in wp["to"] if self.is_waypoint_active(int(t))],
                }
            )

        with open(self.save_name, "w") as f:
            yaml.dump(data_to_save, f, sort_keys=False, default_flow_style=None)

    def save_and_quit(self):
        if self.save():
            self.close()

    def paint_map(self, painter):
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        h, w = self.map_raw.shape[:2]
        target = QRectF(self.offset_x, self.offset_y, w * self.zoom, h * self.zoom)
        painter.drawImage(target, self.map_image)

        self.draw_connections(painter)
        self.draw_waypoints(painter)
        self.draw_drag_arrow(painter)

    def draw_connections(self, painter):
        drawn_bidirectional = set()
        for wp in self.waypoints:
            if wp.get("_deleted", False):
                continue
            sx, sy = self.world_to_pixel(wp["pose"][0], wp["pose"][1])
            start = QPointF(*self.map_to_view(sx, sy))
            for target_id in wp["to"]:
                if self.is_waypoint_active(target_id):
                    target_wp = self.get_waypoint(target_id)
                    if target_wp is None:
                        continue
                    tx, ty = self.world_to_pixel(target_wp["pose"][0], target_wp["pose"][1])
                    end = QPointF(*self.map_to_view(tx, ty))
                    is_bidirectional = wp["id"] in target_wp.get("to", [])
                    if is_bidirectional:
                        pair = tuple(sorted((wp["id"], target_id)))
                        if pair in drawn_bidirectional:
                            continue
                        drawn_bidirectional.add(pair)
                        self.draw_connection_line(painter, start, end, QColor(0, 122, 255), 3, False)
                    else:
                        self.draw_connection_line(painter, start, end, QColor(0, 180, 70), 2, True)

    def draw_connection_line(self, painter, start, end, color, width, arrow):
        pen = QPen(color, width)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(color)
        painter.drawLine(start, end)

        if not arrow:
            return

        dx = end.x() - start.x()
        dy = end.y() - start.y()
        length = math.hypot(dx, dy)
        if length < 18:
            return

        angle = math.atan2(dy, dx)
        head_size = 8
        tip = QPointF(
            end.x() - 8 * math.cos(angle),
            end.y() - 8 * math.sin(angle),
        )
        p1 = QPointF(
            tip.x() - head_size * math.cos(angle - math.pi / 6),
            tip.y() - head_size * math.sin(angle - math.pi / 6),
        )
        p2 = QPointF(
            tip.x() - head_size * math.cos(angle + math.pi / 6),
            tip.y() - head_size * math.sin(angle + math.pi / 6),
        )
        painter.drawPolygon(QPolygonF([tip, p1, p2]))

    def draw_waypoints(self, painter):
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)

        for wp in self.waypoints:
            if wp.get("_deleted", False):
                continue
            px, py = self.world_to_pixel(wp["pose"][0], wp["pose"][1])
            vx, vy = self.map_to_view(px, py)
            point = QPointF(vx, vy)

            if wp["id"] == 0:
                color = QColor(0, 220, 0)
                radius = 7
            elif wp.get("_fixed", False):
                color = QColor(0, 220, 0)
                radius = 7
            elif wp["id"] == self.selected_from:
                color = QColor(0, 170, 255)
                radius = 7
            elif wp["id"] == self.selected_id:
                color = QColor(255, 210, 0)
                radius = 7
            else:
                color = QColor(230, 30, 30)
                radius = 6

            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setBrush(color)
            painter.drawEllipse(point, radius, radius)

            yaw = wp["pose"][2] if len(wp["pose"]) > 2 else 0.0
            arrow_length = max(14, min(34, int(12 * max(1.0, min(self.zoom, 2.5)))))
            end = QPointF(vx + arrow_length * math.cos(yaw), vy - arrow_length * math.sin(yaw))
            self.draw_arrow(painter, point, end, QColor(190, 0, 190), 2, 7)

            text_pos = QPointF(vx + 9, vy - 9)
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.drawText(text_pos, str(wp["id"]))

    def draw_drag_arrow(self, painter):
        if self.drag_start and self.drag_current:
            start = QPointF(*self.map_to_view(*self.drag_start))
            end = QPointF(*self.map_to_view(*self.drag_current))
            self.draw_arrow(painter, start, end, QColor(0, 190, 255), 2, 9)

    def draw_arrow(self, painter, start, end, color, width, head_size):
        pen = QPen(color, width)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(color)
        painter.drawLine(start, end)

        dx = end.x() - start.x()
        dy = end.y() - start.y()
        if math.hypot(dx, dy) < 1.0:
            return

        angle = math.atan2(dy, dx)
        p1 = QPointF(
            end.x() - head_size * math.cos(angle - math.pi / 6),
            end.y() - head_size * math.sin(angle - math.pi / 6),
        )
        p2 = QPointF(
            end.x() - head_size * math.cos(angle + math.pi / 6),
            end.y() - head_size * math.sin(angle + math.pi / 6),
        )
        painter.drawPolygon(QPolygonF([end, p1, p2]))


def default_map_yaml():
    default_map = os.path.join(MAP_DIR, "map_01.yaml")
    if os.path.exists(default_map):
        return default_map
    print(TEXT[initial_language()]["no_map"])
    sys.exit(1)


def resolve_map_arg(map_arg):
    if os.path.isabs(map_arg) or map_arg.endswith(".yaml"):
        return map_arg
    return os.path.join(MAP_DIR, f"{map_arg}.yaml")


def resolve_args(argv):
    if len(argv) < 2:
        return default_map_yaml(), DEFAULT_SAVE_NAME

    if len(argv) == 2:
        arg = argv[1]
        map_yaml = resolve_map_arg(arg)
        if os.path.exists(map_yaml):
            return map_yaml, DEFAULT_SAVE_NAME
        return default_map_yaml(), normalize_save_name(arg)

    return resolve_map_arg(argv[1]), normalize_save_name(argv[2])


if __name__ == "__main__":
    map_yaml, save_name = resolve_args(sys.argv)

    app = QApplication(sys.argv)
    editor = WaypointEditor(map_yaml, save_name)
    editor.show()
    sys.exit(app.exec_())
