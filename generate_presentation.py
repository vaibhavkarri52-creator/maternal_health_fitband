import sys
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_deck(filename="Maternal_Health_Fitband_Presentation.pptx"):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Classic refined human palette
    NAVY = RGBColor(24, 43, 73)          # Dark classic navy for titles & headers
    CHARCOAL = RGBColor(40, 44, 52)      # Soft black / charcoal for body text
    MUTED_GRAY = RGBColor(108, 117, 125) # Professional gray for sub-elements
    CARD_BG = RGBColor(248, 250, 252)    # Subtle off-white card fill
    BORDER_COL = RGBColor(218, 224, 233) # Subtle neat border
    ACCENT_LINE = RGBColor(41, 70, 115)  # Navy accent for section tracker

    slides_data = [
        # Slide 1: Title
        {
            "type": "title",
            "title": "Maternal Health Fitband",
            "subtitle": "Continuous Physiological Monitoring and Risk Screening Using Embedded TinyML",
            "meta": "Academic & Clinical Project Presentation"
        },
        # Slide 2: Introduction
        {
            "type": "content",
            "section": "INTRODUCTION",
            "title": "Maternal Healthcare Context",
            "subtitle": "Overview of prenatal vital signs and remote monitoring needs",
            "points": [
                ("Maternal Well-being Priority: ", "Maternal health during pregnancy demands frequent, reliable tracking of blood pressure, heart rate, and metabolic stability."),
                ("Vulnerability Windows: ", "Complications such as gestational hypertension and gestational diabetes often develop silently between routine clinic visits."),
                ("Wearable Sensor Shift: ", "Low-cost non-invasive wearable sensors enable continuous physiological monitoring outside traditional hospital settings."),
                ("Project Scope: ", "This project presents a wrist-worn band combining optical PPG and motion sensing with on-device machine learning for maternal care.")
            ]
        },
        # Slide 3: Background
        {
            "type": "content",
            "section": "BACKGROUND",
            "title": "Physiological Parameters in Pregnancy",
            "subtitle": "Hemodynamic changes and clinical standards during gestation",
            "points": [
                ("Hemodynamic Changes: ", "Maternal blood volume and cardiac output rise substantially across trimesters, altering baseline cardiovascular metrics."),
                ("Clinical Standards: ", "ACOG and WHO define strict screening criteria for gestational hypertension (SBP ≥ 140 or DBP ≥ 90 mmHg)."),
                ("Optical PPG Principles: ", "Photoplethysmography measures pulse waveform characteristics, blood volume changes, and blood oxygen saturation (SpO2)."),
                ("Inertial Motion Tracking: ", "6-axis accelerometer and gyroscope tracking captures maternal physical activity levels and posture during rest.")
            ]
        },
        # Slide 4: Research Gap
        {
            "type": "content",
            "section": "RESEARCH GAP",
            "title": "Limitations in Current Solutions",
            "subtitle": "Why existing consumer wearables fall short for pregnant mothers",
            "points": [
                ("General vs Pregnancy Baselines: ", "Most smartbands are calibrated for general adults and ignore trimester-specific cardiovascular changes."),
                ("Intermittent Cuff Dependency: ", "Standard blood pressure cuffs cause sleep disturbance and fail to detect acute daytime pressure spikes."),
                ("Postural Risk Blindness: ", "Conventional trackers do not monitor maternal supine sleeping posture, a known risk factor for inferior vena cava compression."),
                ("Cloud Dependency & Latency: ", "Many remote health devices rely entirely on cloud servers, risking missed warnings in low-connectivity rural settings.")
            ]
        },
        # Slide 5: Motivation
        {
            "type": "content",
            "section": "MOTIVATION",
            "title": "Project Rationale",
            "subtitle": "Driving factors behind an integrated maternal monitoring system",
            "points": [
                ("Preventing Complications: ", "Early detection of preeclampsia and glycemic anomalies can prevent emergency hospital admissions."),
                ("Affordable Hardware: ", "Using accessible components (ESP32, MAX30102, MPU6050) brings continuous care within reach for low-resource settings."),
                ("Immediate Edge Feedback: ", "Running inference directly on the microcontroller delivers immediate alerts without internet connectivity."),
                ("Clinical Reassurance: ", "Continuous tracking provides objective daily trends for obstetricians and reassurance for expectant mothers.")
            ]
        },
        # Slide 6: Objectives
        {
            "type": "content",
            "section": "OBJECTIVES",
            "title": "Core Technical Goals",
            "subtitle": "Specific targets established for hardware and software design",
            "points": [
                ("Design Embedded Hardware: ", "Construct a compact wrist unit pairing the ESP32 microcontroller with optical PPG and 6-axis IMU sensors."),
                ("Develop TinyML Inference: ", "Port trained clinical estimation models into optimized C++ headers executing on-chip in under 3 ms."),
                ("Implement Multi-Parameter Tracking: ", "Estimate heart rate, blood pressure, non-invasive glucose trend, activity, and sleep posture."),
                ("Deliver a Web Dashboard: ", "Provide a responsive local web interface and BLE receiver for live visualization and clinical alerting.")
            ]
        },
        # Slide 7: Work Done
        {
            "type": "content",
            "section": "WORK DONE",
            "title": "Implemented System Components",
            "subtitle": "Summary of hardware assembly, firmware, and software engineering",
            "points": [
                ("Shared I2C Hardware Bus: ", "Wired the MAX30102 optical sensor and MPU6050 6-axis IMU to ESP32 hardware pins GPIO 21 and 22."),
                ("Production Firmware: ", "Authored modular C++ firmware supporting live JSON telemetry over USB serial and Bluetooth Low Energy."),
                ("Embedded TinyML Pipeline: ", "Created maternal_models_esp32.h for instantaneous edge classification of risk, posture, and activity."),
                ("Full-Stack Companion Platform: ", "Built a Tornado-powered web application with Server-Sent Events, ML inference engine, and clinical rules.")
            ]
        },
        # Slide 8: Methodology - Architecture
        {
            "type": "content",
            "section": "METHODOLOGY",
            "title": "System Architecture & Data Flow",
            "subtitle": "End-to-end pipeline from physical sensors to user interface",
            "points": [
                ("Sensor Acquisition: ", "MAX30102 samples Red and IR channels while MPU6050 captures tri-axial acceleration and rotational velocity."),
                ("Signal Conditioning: ", "Bandpass filters eliminate motion artifacts; the Beer-Lambert ratio-of-ratios extracts SpO2 and pulse waveforms."),
                ("Dual-Stage Inference: ", "Lightweight TinyML runs directly on the microcontroller; full scikit-learn models run within the companion server."),
                ("Clinical Rule Mapping: ", "Model predictions pass through deterministic ACOG and NICE guideline thresholds before alerting.")
            ]
        },
        # Slide 9: Methodology - ML Models
        {
            "type": "content",
            "section": "METHODOLOGY",
            "title": "Machine Learning Pipeline",
            "subtitle": "Model selection and parameter estimation pipeline",
            "points": [
                ("Cardiovascular Estimation: ", "Random Forest and Gradient Boosting regressors estimate SBP, DBP, and motion-corrected heart rate."),
                ("Metabolic & Glycemic Screening: ", "A 3-tier risk classifier maps optical pulse morphology and meal context to glycemic state."),
                ("Biomechanical Classification: ", "IMU motion metrics classify maternal states into sedentary, walking, light exercise, and overexertion."),
                ("Nocturnal Posture Staging: ", "Gravitational vector angles identify supine sleeping to trigger gentle positional change alerts.")
            ]
        },
        # Slide 10: Results - Hardware & Edge
        {
            "type": "content",
            "section": "RESULTS",
            "title": "Hardware & Firmware Performance",
            "subtitle": "Measured operational metrics and edge benchmark results",
            "points": [
                ("Rapid On-Chip Inference: ", "Embedded TinyML routines execute complete feature evaluation on the ESP32 in less than 2.5 milliseconds."),
                ("Reliable Data Transmission: ", "Standardized JSON packets stream continuously at 115200 baud over serial and over BLE advertising."),
                ("Low Power Footprint: ", "Operational current draw remains within typical wearable power budgets across active sensing modes."),
                ("Stable Optical Tracking: ", "Adaptive peak detection reliably extracts pulse amplitudes even during moderate maternal arm movements.")
            ]
        },
        # Slide 11: Results - Model Performance
        {
            "type": "content",
            "section": "RESULTS",
            "title": "Analytical Model Performance",
            "subtitle": "Validation results across maternal vital metrics",
            "points": [
                ("Blood Pressure Accuracy: ", "Estimated SBP and DBP track clinical references closely within accepted ambulatory error margins."),
                ("Postural Classification: ", "Supine versus lateral sleeping posture detection achieved high sensitivity using pitch and roll tilt angles."),
                ("Motion Robustness: ", "IMU-assisted heart rate filtering maintained pulse tracking integrity during active maternal walking."),
                ("Deterministic Alert Reliability: ", "ACOG rule checks operated without false alerts across verified normal maternal physiological profiles.")
            ]
        },
        # Slide 12: Discussion
        {
            "type": "content",
            "section": "DISCUSSION",
            "title": "Findings & Practical Considerations",
            "subtitle": "Analysis of trade-offs, strengths, and deployment factors",
            "points": [
                ("Edge vs Cloud Balance: ", "Local TinyML ensures safety-critical alerts trigger instantly, while web dashboards offer richer historical views."),
                ("Clinical Guideline Alignment: ", "Restricting alerting logic to deterministic ACOG rules prevents erratic predictions common in pure neural nets."),
                ("Prototype Considerations: ", "Optical estimations serve as an early screening tool and require individual baseline calibration."),
                ("Maternal Ergonomics: ", "Wrist placement offers high compliance, though optical skin contact must remain snug for signal quality.")
            ]
        },
        # Slide 13: Conclusion
        {
            "type": "content",
            "section": "CONCLUSION",
            "title": "Summary of Contributions",
            "subtitle": "Key outcomes and practical benefits of the developed system",
            "points": [
                ("Integrated Prototype: ", "Delivered a functional wearable monitoring system combining low-cost sensors with embedded intelligence."),
                ("Dual-Level Intelligence: ", "Demonstrated fast on-chip TinyML inference alongside a comprehensive companion clinical server."),
                ("Specialized Care Focus: ", "Addressed pregnancy-specific risks including aortocaval compression and gestational hypertension trends."),
                ("Future Scope: ", "Future enhancements include clinical trial validation, custom flexible enclosure design, and low-power battery optimization.")
            ]
        },
        # Slide 14: References
        {
            "type": "content",
            "section": "REFERENCES",
            "title": "Key Clinical & Technical Standards",
            "subtitle": "Primary publications and medical guidelines referenced",
            "points": [
                ("ACOG Practice Bulletin No. 222: ", "Gestational Hypertension and Preeclampsia, American College of Obstetricians and Gynecologists, 2020."),
                ("NICE Guideline NG133: ", "Hypertension in pregnancy: diagnosis and management, National Institute for Health and Care Excellence, 2019."),
                ("WHO Standards: ", "WHO recommendations on antenatal care for a positive pregnancy experience, World Health Organization, 2016."),
                ("Allen, J. (2007): ", "Photoplethysmography and its application in clinical physiological measurement, Physiological Measurement, 28(3).")
            ]
        }
    ]

    for item in slides_data:
        slide = prs.slides.add_slide(blank_layout)

        if item["type"] == "title":
            # Slide Background
            bg_box = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5)
            )
            bg_box.fill.solid()
            bg_box.fill.fore_color.rgb = RGBColor(250, 251, 252)
            bg_box.line.fill.background()

            # Center Card Container
            card = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.5), Inches(1.3), Inches(10.333), Inches(4.9)
            )
            card.fill.solid()
            card.fill.fore_color.rgb = RGBColor(255, 255, 255)
            card.line.color.rgb = BORDER_COL
            card.line.width = Pt(1.5)

            # Slim decorative accent bar
            bar = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE, Inches(2.2), Inches(2.1), Inches(1.2), Inches(0.08)
            )
            bar.fill.solid()
            bar.fill.fore_color.rgb = NAVY
            bar.line.fill.background()

            # Title
            tx = slide.shapes.add_textbox(Inches(2.1), Inches(2.4), Inches(9.1), Inches(1.4))
            tf = tx.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
            p = tf.paragraphs[0]
            p.text = item["title"]
            p.font.name = "Georgia"
            p.font.size = Pt(40)
            p.font.bold = True
            p.font.color.rgb = NAVY

            # Subtitle
            tx_sub = slide.shapes.add_textbox(Inches(2.1), Inches(4.0), Inches(9.1), Inches(0.9))
            tf_sub = tx_sub.text_frame
            tf_sub.word_wrap = True
            tf_sub.margin_left = tf_sub.margin_top = tf_sub.margin_right = tf_sub.margin_bottom = 0
            p_sub = tf_sub.paragraphs[0]
            p_sub.text = item["subtitle"]
            p_sub.font.name = "Arial"
            p_sub.font.size = Pt(17)
            p_sub.font.color.rgb = CHARCOAL

            # Meta Text
            tx_meta = slide.shapes.add_textbox(Inches(2.1), Inches(5.1), Inches(9.1), Inches(0.5))
            tf_meta = tx_meta.text_frame
            tf_meta.word_wrap = True
            tf_meta.margin_left = tf_meta.margin_top = tf_meta.margin_right = tf_meta.margin_bottom = 0
            p_meta = tf_meta.paragraphs[0]
            p_meta.text = item["meta"]
            p_meta.font.name = "Arial"
            p_meta.font.size = Pt(13)
            p_meta.font.color.rgb = MUTED_GRAY

        else:
            # Header section tracker
            tx_sec = slide.shapes.add_textbox(Inches(1.0), Inches(0.55), Inches(11.333), Inches(0.35))
            tf_sec = tx_sec.text_frame
            tf_sec.margin_left = tf_sec.margin_top = tf_sec.margin_right = tf_sec.margin_bottom = 0
            p_sec = tf_sec.paragraphs[0]
            p_sec.text = item["section"]
            p_sec.font.name = "Arial"
            p_sec.font.size = Pt(11)
            p_sec.font.bold = True
            p_sec.font.color.rgb = ACCENT_LINE

            # Main Slide Title
            tx_title = slide.shapes.add_textbox(Inches(1.0), Inches(0.9), Inches(11.333), Inches(0.6))
            tf_title = tx_title.text_frame
            tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
            p_title = tf_title.paragraphs[0]
            p_title.text = item["title"]
            p_title.font.name = "Georgia"
            p_title.font.size = Pt(26)
            p_title.font.bold = True
            p_title.font.color.rgb = NAVY

            # Subtitle / context description
            tx_sub = slide.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.333), Inches(0.35))
            tf_sub = tx_sub.text_frame
            tf_sub.margin_left = tf_sub.margin_top = tf_sub.margin_right = tf_sub.margin_bottom = 0
            p_sub = tf_sub.paragraphs[0]
            p_sub.text = item["subtitle"]
            p_sub.font.name = "Arial"
            p_sub.font.size = Pt(13)
            p_sub.font.color.rgb = MUTED_GRAY

            # Horizontal divider line
            line = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(1.95), Inches(11.333), Inches(0.02)
            )
            line.fill.solid()
            line.fill.fore_color.rgb = BORDER_COL
            line.line.fill.background()

            # Main content card container
            body_box = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(2.2), Inches(11.333), Inches(4.55)
            )
            body_box.fill.solid()
            body_box.fill.fore_color.rgb = CARD_BG
            body_box.line.color.rgb = BORDER_COL
            body_box.line.width = Pt(1)

            # Single unified text frame with paragraph spacing for clean layout
            tx_points = slide.shapes.add_textbox(Inches(1.4), Inches(2.4), Inches(10.533), Inches(4.15))
            tf_points = tx_points.text_frame
            tf_points.word_wrap = True
            tf_points.margin_left = tf_points.margin_top = tf_points.margin_right = tf_points.margin_bottom = 0

            pts = item["points"]
            for i, (lead, body) in enumerate(pts):
                p_pt = tf_points.paragraphs[0] if i == 0 else tf_points.add_paragraph()
                
                # Spacing between points
                p_pt.space_after = Pt(18)
                p_pt.line_spacing = 1.25

                # Bullet mark
                bullet_run = p_pt.add_run()
                bullet_run.text = "■  "
                bullet_run.font.name = "Arial"
                bullet_run.font.size = Pt(12)
                bullet_run.font.color.rgb = NAVY

                # Bold Lead
                lead_run = p_pt.add_run()
                lead_run.text = lead
                lead_run.font.name = "Arial"
                lead_run.font.size = Pt(15)
                lead_run.font.bold = True
                lead_run.font.color.rgb = NAVY

                # Normal English Body
                body_run = p_pt.add_run()
                body_run.text = body
                body_run.font.name = "Arial"
                body_run.font.size = Pt(15)
                body_run.font.bold = False
                body_run.font.color.rgb = CHARCOAL

            # Subtle slide footer
            tx_foot = slide.shapes.add_textbox(Inches(1.0), Inches(6.9), Inches(11.333), Inches(0.3))
            tf_foot = tx_foot.text_frame
            tf_foot.margin_left = tf_foot.margin_top = tf_foot.margin_right = tf_foot.margin_bottom = 0
            p_foot = tf_foot.paragraphs[0]
            p_foot.text = "Maternal Health Fitband  |  Clinical IoT & TinyML Project"
            p_foot.font.name = "Arial"
            p_foot.font.size = Pt(10)
            p_foot.font.color.rgb = MUTED_GRAY

    prs.save(filename)
    print(f"Presentation regenerated successfully at {filename}")

if __name__ == "__main__":
    create_deck()
