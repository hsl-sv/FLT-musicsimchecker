import os
import glob
import pathlib
from pathlib import Path
import numpy as np

from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget, QSizePolicy, QApplication
from PySide6.QtGui import QStandardItem, QStandardItemModel, QBrush, QColor, QImage, QPixmap, QPainter, QPen

from process import analyze_1_to_n_match_parallel

class UIManager(object):
    
    def __init__(self, MainWindow):
        super().__init__()
        self.cwd = os.getcwd()
        self.cwd_filelist = []
        self.cwd_currentfile = None
        self.exif = None
        self.items = QStandardItemModel()
        self.audio_ext = ('.flac', '.mp3', '.wav')
        self.MainWindow = MainWindow

    def openfile(self):
        file, _ = QFileDialog.getOpenFileName(None, "Select Source File", self.cwd, "Audio Files (*.flac *.mp3 *.wav)")
        if file:
            self.cwd_currentfile = file
            self.MainWindow.labelSourceFile.setText(file)
            self.current_file = file

    def opendialog(self):
        folder = QFileDialog.getExistingDirectory(None, "Select Directory", self.cwd)
        self.cwd = folder
        self.populate_listview(self.cwd)

    def populate_listview(self, folder):
        flist = []
        
        folder_path = Path(folder)

        # 2. 하위 모든 폴더를 1번만 스캔하며 오디오 파일만 추출
        flist = [
            str(p) for p in folder_path.rglob("*") 
            if p.is_file() and p.suffix.lower() in self.audio_ext
        ]
        
        dirlist = []
        self.items.clear()
        for i, fpath in enumerate(flist):
            self.items.appendRow(QStandardItem(fpath))
        
        self.cwd_filelist = flist
        self.MainWindow.listViewTargetFolder.setModel(self.items)

    def start(self):
        if not self.cwd_filelist:
            QMessageBox.warning(None, "Warning", "No files to process.")
            return
        source_file = self.MainWindow.labelSourceFile.text()
        if not source_file or not os.path.isfile(source_file):
            QMessageBox.warning(None, "Warning", "Please select a valid source file.")
            return

        cost_threshold = 0.15
        min_coverage = 0.50
        results = analyze_1_to_n_match_parallel(source_file, self.cwd_filelist, cost_threshold=cost_threshold, min_coverage=min_coverage)


        GREEN_START = '<span style="color: #2ecc71; font-weight: bold;">' # 초록색
        RED_START = '<span style="color: #e74c3c; font-weight: bold;">'   # 빨간색
        SPAN_END = '</span>'

        lines = []

        for rank, res in enumerate(results, 1):
            if res.get("status") == "success":
                # MATCH 여부에 따라 태그 부분만 HTML로 감싸기
                if res["is_match"]:
                    status_tag = f"{GREEN_START}[★ MATCH DETECTED]{SPAN_END}"
                else:
                    status_tag = f"{RED_START}[X NO MATCH]{SPAN_END}"

                lines.append(f"#{rank:<2} {status_tag} | {res['filename_B']}")
                lines.append(f"     └─ Cost: {res['cost']:.4f} (기준 <= {cost_threshold}) | 피치 변조: {res['pitch_msg']}")
                lines.append(f"     └─ 커버리지: {res['coverage_pct']:.1f}% (기준 >= {min_coverage*100:.0f}%) | 기울기: {res['slope']:.2f}")
                lines.append(f"     └─ 탐색 구간: {res['start_pct']:.2f}% ~ {res['end_pct']:.2f}% ({res['start_time']:.1f}초 ~ {res['end_time']:.1f}초 / 총 {res['dur_B']:.1f}초)")
            else:
                # 에러 발생 시 줄 전체를 빨간색으로 처리
                lines.append(f"{RED_START}#{rank:<2} [ERROR] | {res['filename_B']} - {res['status']}{SPAN_END}")
            lines.append("-" * 75)

        # 2. QTextEdit에 뿌려줄 때 개행 문자(\n)와 공백을 그대로 유지하기 위해 <pre> 태그 사용
        # (self.textEdit 변수명은 본인 코드에 맞게 수정하세요)
        full_text = "\n".join(lines)
        self.MainWindow.textEditResult.setHtml(f"<pre style='font-family: Consolas, Monospace; font-size: 10pt;'>{full_text}</pre>")