from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(r"C:\neuralangelo")
OUT_DIR = ROOT / "output" / "pdf"
OUT_PATH = OUT_DIR / "colmap_principle_steps_2356.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\msjh.ttc")


def register_fonts():
    pdfmetrics.registerFont(TTFont("MSJH", str(FONT_PATH)))
    pdfmetrics.registerFont(TTFont("MSJH-Bold", str(FONT_PATH)))


def make_styles():
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=styles["Title"],
            fontName="MSJH-Bold",
            fontSize=22,
            leading=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17324D"),
            wordWrap="CJK",
            spaceAfter=7 * mm,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=styles["BodyText"],
            fontName="MSJH",
            fontSize=10.5,
            leading=17,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4A5568"),
            wordWrap="CJK",
            spaceAfter=8 * mm,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=styles["Heading1"],
            fontName="MSJH-Bold",
            fontSize=15,
            leading=22,
            textColor=colors.HexColor("#12385B"),
            wordWrap="CJK",
            spaceBefore=3 * mm,
            spaceAfter=3 * mm,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=styles["Heading2"],
            fontName="MSJH-Bold",
            fontSize=12,
            leading=18,
            textColor=colors.HexColor("#1E5A8A"),
            wordWrap="CJK",
            spaceBefore=2 * mm,
            spaceAfter=1.5 * mm,
        ),
        "body": ParagraphStyle(
            "body",
            parent=styles["BodyText"],
            fontName="MSJH",
            fontSize=10.4,
            leading=17,
            textColor=colors.HexColor("#1F2933"),
            wordWrap="CJK",
            spaceAfter=2.4 * mm,
        ),
        "small": ParagraphStyle(
            "small",
            parent=styles["BodyText"],
            fontName="MSJH",
            fontSize=8.6,
            leading=13,
            textColor=colors.HexColor("#4A5568"),
            wordWrap="CJK",
        ),
        "code": ParagraphStyle(
            "code",
            parent=styles["Code"],
            fontName="MSJH",
            fontSize=8.6,
            leading=12,
            textColor=colors.HexColor("#243B53"),
            backColor=colors.HexColor("#F5F7FA"),
            borderPadding=5,
            wordWrap="CJK",
            spaceAfter=2.5 * mm,
        ),
    }


def p(text, style):
    return Paragraph(text, style)


def bullet(text, style):
    return Paragraph(f"• {text}", style)


def source_table(styles):
    rows = [
        ["流程啟動", "projects/neuralangelo/scripts/run_colmap.sh"],
        ["Step 2 讀圖與 SIFT", "src/base/image_reader.h、src/feature/extraction.h、src/feature/sift.h"],
        ["Step 3 配對", "src/feature/matching.h、src/feature/sift.cc"],
        ["Step 5 SfM / Pose", "src/sfm/incremental_mapper.h、src/sfm/incremental_mapper.cc、src/estimators/pose.h"],
        ["Step 6 三角化", "src/sfm/incremental_triangulator.cc、src/estimators/triangulation.h"],
        ["輸出座標轉換", "projects/neuralangelo/scripts/convert_data_to_json.py"],
    ]
    data = [[p(a, styles["small"]), p(b, styles["small"])] for a, b in rows]
    table = Table(data, colWidths=[34 * mm, 116 * mm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("MSJH", 8)
    canvas.setFillColor(colors.HexColor("#718096"))
    canvas.drawRightString(A4[0] - 18 * mm, 11 * mm, f"{doc.page}")
    canvas.drawString(18 * mm, 11 * mm, "COLMAP 原理導向整理 - 根據本機程式碼")
    canvas.restoreState()


def build_pdf():
    register_fonts()
    styles = make_styles()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = Frame(18 * mm, 18 * mm, A4[0] - 36 * mm, A4[1] - 36 * mm, id="normal")
    doc = BaseDocTemplate(
        str(OUT_PATH),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=add_page_number)])

    story = []
    story.append(p("COLMAP SfM 原理導向整理", styles["title"]))
    story.append(
        p(
            "內容根據 C:\\neuralangelo 內的 COLMAP / Neuralangelo 程式碼整理。"
            "重點放在 Step 2、Step 3、Step 5、Step 6 的運行原理；參數只作為輔助理解。",
            styles["subtitle"],
        )
    )

    story.append(p("整體核心原理", styles["h1"]))
    story.append(
        p(
            "COLMAP 在這個專案中的流程是 feature_extractor → sequential_matcher → mapper → image_undistorter。"
            "真正與 pose 估計最直接相關的是前三個階段：先找特徵、再找影像間對應，最後由 mapper 做增量式 SfM。",
            styles["body"],
        )
    )
    story.append(
        p(
            "SfM 的核心想法是：如果不同照片中有些 2D 特徵點其實來自同一個真實場景點，"
            "那就可以利用相機投影幾何，同時推回相機的位置與方向，也就是 pose，以及場景中的 3D 點位置。",
            styles["body"],
        )
    )
    story.append(
        p(
            "因此 COLMAP 不是神經網路。它不是用訓練資料學出一個模型來預測 pose，而是根據特徵匹配、"
            "幾何驗證、RANSAC、三角化與 Bundle Adjustment 逐步求解。",
            styles["body"],
        )
    )
    story.append(p("本 PDF 使用的程式碼來源", styles["h2"]))
    story.append(source_table(styles))
    story.append(PageBreak())

    story.append(p("Step 2：SIFT 特徵點 - 先找可重複辨識的局部點", styles["h1"]))
    story.append(
        p(
            "這一步由 run_colmap.sh 呼叫 colmap feature_extractor。程式設定 image_path 為 images_raw，"
            "把輸入圖片交給 ImageReader，再由 SiftFeatureExtractor 產生 keypoints 與 descriptors，並寫入 database.db。",
            styles["body"],
        )
    )
    story.append(p("原理", styles["h2"]))
    story.append(
        p(
            "一張圖片有大量像素，但不是所有像素都適合拿來配對。像白牆、天空或模糊區域，"
            "即使在下一張圖中也很難穩定找到同一個位置。SIFT 的目的就是先挑出比較穩定、"
            "紋理明顯、在尺度或角度變化下仍可能被再次辨識的局部特徵。",
            styles["body"],
        )
    )
    story.append(
        p(
            "每個被保留下來的特徵點包含兩類資訊：keypoint 表示它在影像中的位置與局部尺度等幾何資訊；"
            "descriptor 則是描述該點周圍影像紋理的數值向量。後面的 matching 主要依靠 descriptor 來判斷兩個點是否相似。",
            styles["body"],
        )
    )
    story.append(p("參數如何輔助理解", styles["h2"]))
    story.append(bullet("camera_model=SIMPLE_RADIAL：表示此流程使用簡單徑向畸變相機模型。", styles["body"]))
    story.append(bullet("single_camera=true：代表所有影像共用同一組相機內參，符合單一相機連續拍攝的假設。", styles["body"]))
    story.append(bullet("max_num_features=8192：限制每張影像最多保留多少 SIFT 特徵。", styles["body"]))
    story.append(bullet("peak_threshold：控制特徵點強度門檻，太弱的點不容易被保留。", styles["body"]))
    story.append(bullet("edge_threshold：排除過度像邊緣、定位不穩定的點。", styles["body"]))
    story.append(
        p(
            "報告說法：Step 2 的目的不是估 pose，而是先把每張影像轉成一組穩定的局部特徵。"
            "後面的流程不直接比對整張圖片，而是比對這些特徵點與 descriptor。",
            styles["code"],
        )
    )

    story.append(p("Step 3：Feature Matching - 找出不同圖片中的同一個場景點", styles["h1"]))
    story.append(
        p(
            "這一步由 run_colmap.sh 呼叫 colmap sequential_matcher。它會從 database.db 讀取 Step 2 產生的 descriptors，"
            "並把匹配結果與兩視角幾何驗證結果寫回資料庫。",
            styles["body"],
        )
    )
    story.append(p("原理", styles["h2"]))
    story.append(
        p(
            "如果兩張影像中的 descriptor 很相似，COLMAP 會先認為這兩個 2D 特徵點可能是同一個真實 3D 點"
            "從不同相機位置投影到影像上的結果。這形成的是 2D-2D 對應。",
            styles["body"],
        )
    )
    story.append(
        p(
            "但是 descriptor 相似不代表一定正確。例如重複紋理、相似窗框或相似表面都可能造成錯誤匹配。"
            "所以程式中 matching options 包含 verify_matches，會進一步做幾何驗證，保留符合兩張影像相機幾何關係的匹配。",
            styles["body"],
        )
    )
    story.append(p("參數如何輔助理解", styles["h2"]))
    story.append(bullet("sequential_matcher：表示資料被當作連續影像處理，適合影片 frame 或依時間排序的圖片。", styles["body"]))
    story.append(bullet("overlap=10：預設優先讓每張影像和附近影像做配對。", styles["body"]))
    story.append(bullet("quadratic_overlap=true：會額外加入一些更遠但仍合理的序列影像配對。", styles["body"]))
    story.append(bullet("verify_matches=true：配對後做幾何驗證，刪除不合理的匹配。", styles["body"]))
    story.append(
        p(
            "報告說法：Step 3 是把不同圖片中的特徵點連起來。它先用 descriptor 相似度找候選匹配，"
            "再用幾何驗證過濾錯誤，留下較可信的 2D-2D 對應。",
            styles["code"],
        )
    )

    story.append(PageBreak())
    story.append(p("Step 5：SfM 估計 Pose - 用 2D-3D 對應反推相機位置", styles["h1"]))
    story.append(
        p(
            "這一步由 run_colmap.sh 呼叫 colmap mapper。mapper 使用增量式 SfM：先建立初始模型，"
            "再逐張加入新的影像。程式中的 RegisterNextImage 會替新影像估計 Qvec 與 Tvec。",
            styles["body"],
        )
    )
    story.append(p("原理", styles["h2"]))
    story.append(
        p(
            "當 COLMAP 已經有部分 3D 點後，新影像如果也看到了這些點，就能建立 2D-3D 對應："
            "2D 是新影像上的特徵點位置，3D 是 reconstruction 中已經存在的空間點。",
            styles["body"],
        )
    )
    story.append(
        p(
            "接著程式呼叫 EstimateAbsolutePose，輸入 tri_points2D 與 tri_points3D，輸出 image.Qvec() 與 image.Tvec()。"
            "直覺上，這是在問：相機要站在哪裡、朝哪個方向看，才會讓這些已知 3D 點投影到影像中的這些 2D 位置？",
            styles["body"],
        )
    )
    story.append(
        p(
            "因為 2D-3D 對應中可能有錯誤匹配，所以程式使用 RANSAC。RANSAC 會反覆抽樣估 pose，"
            "再檢查哪些對應點在這個 pose 下投影誤差合理。這些合理的點稱為 inliers。"
            "如果 inliers 足夠，才接受這張影像的 pose，接著再用 RefineAbsolutePose 做細部優化。",
            styles["body"],
        )
    )
    story.append(p("參數如何輔助理解", styles["h2"]))
    story.append(bullet("init_min_num_inliers=100：初始化模型時，需要足夠多可靠匹配。", styles["body"]))
    story.append(bullet("init_max_error=4.0：初始化時接受的最大重投影誤差，單位是 pixel。", styles["body"]))
    story.append(bullet("abs_pose_min_num_inliers=30：新影像估 pose 時，至少要有 30 個可信 2D-3D 對應。", styles["body"]))
    story.append(bullet("abs_pose_max_error=12.0：估 pose 時允許的最大重投影誤差。", styles["body"]))
    story.append(bullet("local_ba_num_images=6：局部 Bundle Adjustment 會優化附近影像，使 pose 與 3D 點更一致。", styles["body"]))
    story.append(
        p(
            "報告說法：Step 5 是 pose 估計的核心。COLMAP 不是直接猜相機 pose，而是利用已知 3D 點"
            "和新影像上的 2D 特徵點，根據投影幾何反推相機的旋轉 Qvec 與平移 Tvec。",
            styles["code"],
        )
    )

    story.append(p("Step 6：Triangulation - 用多張影像交會出 3D 點", styles["h1"]))
    story.append(
        p(
            "在 mapper 中，RegisterNextImage 完成後會呼叫 TriangulateImage。實際三角化邏輯位於"
            " incremental_triangulator.cc 與 triangulation.h。",
            styles["body"],
        )
    )
    story.append(p("原理", styles["h2"]))
    story.append(
        p(
            "單張圖片只能告訴我們某個點位於相機視線的哪個方向，不能直接知道深度。"
            "如果同一個特徵點被兩張或多張已知 pose 的相機看到，就能從每個相機中心往該 2D 點方向拉出射線。",
            styles["body"],
        )
    )
    story.append(
        p(
            "理想情況下，這些射線會在 3D 空間中交會；實際資料有誤差，所以 COLMAP 會找出最合理的 3D 位置，"
            "並根據角度誤差、三角化角度與 cheirality 檢查結果是否可靠。cheirality 的意思是點必須在相機前方。",
            styles["body"],
        )
    )
    story.append(
        p(
            "新三角化出的 3D 點會被加入 reconstruction。這些新 3D 點又能在後續 Step 5 中幫助更多影像估 pose，"
            "因此 SfM 是 pose 與 3D points 互相推進的循環。",
            styles["body"],
        )
    )
    story.append(p("參數如何輔助理解", styles["h2"]))
    story.append(bullet("至少 2 個 observations：至少要兩張影像看到同一點，才能從視線交會估深度。", styles["body"]))
    story.append(bullet("min_tri_angle：兩個視角夾角要夠大，否則深度估計不穩定。", styles["body"]))
    story.append(bullet("residual_type=ANGULAR_ERROR：用視線角度誤差衡量三角化品質。", styles["body"]))
    story.append(bullet("RANSAC max_num_trials=10000：在可能含錯誤 observation 的情況下，多次嘗試找可靠 3D 點。", styles["body"]))
    story.append(
        p(
            "報告說法：Step 6 是把 2D 匹配轉成 3D 點。當多張已知 pose 的相機看到同一個特徵點時，"
            "COLMAP 會利用視線交會的幾何關係計算該點的 3D 座標。",
            styles["code"],
        )
    )

    story.append(PageBreak())
    story.append(p("報告用總結", styles["h1"]))
    story.append(
        p(
            "COLMAP 的 SfM 是一個增量式幾何重建流程。Step 2 先用 SIFT 找出每張影像中穩定的局部特徵；"
            "Step 3 再用 descriptor 相似度與幾何驗證找出不同影像中的同一個場景點；"
            "Step 5 使用已知 3D 點與新影像 2D 點的對應，透過 RANSAC 與 pose refinement 估計相機的 Qvec 與 Tvec；"
            "Step 6 則利用多張已知 pose 的影像，將匹配點三角化成新的 3D 點。",
            styles["body"],
        )
    )
    story.append(
        p(
            "這些步驟不是單向流程，而是互相支援：pose 估得越多，能三角化的 3D 點越多；"
            "3D 點越多，後續影像又越容易估 pose。最後透過 Bundle Adjustment 讓相機 pose 與 3D 點整體更一致。",
            styles["body"],
        )
    )
    story.append(p("補充：Neuralangelo 使用的 pose 輸出", styles["h1"]))
    story.append(
        p(
            "COLMAP 內部的影像 pose 使用 world-to-camera 形式，包含旋轉 qvec 與平移 tvec。"
            "在 Neuralangelo 的 convert_data_to_json.py 中，程式會先由 qvec 得到 rotation，組成 w2c 矩陣，"
            "再取 inverse 得到 c2w，最後透過 _cv_to_gl 轉成 iNGP / OpenGL convention 後寫入 transforms.json。",
            styles["body"],
        )
    )

    doc.build(story)
    print(OUT_PATH)


if __name__ == "__main__":
    build_pdf()
