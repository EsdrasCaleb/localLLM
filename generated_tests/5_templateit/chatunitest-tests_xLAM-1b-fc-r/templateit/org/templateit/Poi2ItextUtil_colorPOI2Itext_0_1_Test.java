package org.templateit;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.awt.Color;
import java.awt.font.FontRenderContext;
import java.awt.font.TextAttribute;
import java.awt.font.TextLayout;
import java.text.AttributedString;
import org.apache.log4j.Logger;
import org.apache.poi.hssf.usermodel.HSSFCell;
import org.apache.poi.hssf.usermodel.HSSFCellStyle;
import org.apache.poi.hssf.usermodel.HSSFFont;
import org.apache.poi.hssf.usermodel.HSSFFormulaEvaluator;
import org.apache.poi.hssf.usermodel.HSSFPalette;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.apache.poi.hssf.util.HSSFColor;
import com.lowagie.text.Element;
import com.lowagie.text.Font;
import com.lowagie.text.Rectangle;
import com.lowagie.text.pdf.PdfPCell;

class Poi2ItextUtil_colorPOI2Itext_0_1_Test {

    @Test
    void colorPOI2Itext() {
        HSSFColor mockColor = Mockito.mock(HSSFColor.class);
        Mockito.when(mockColor.getTriplet()).thenReturn(new short[] { 1, 2, 3 });
        Color expectedColor = new Color(1, 2, 3);
        Color actualColor = Poi2ItextUtil.colorPOI2Itext(mockColor);
        assertEquals(expectedColor, actualColor);
    }
}
