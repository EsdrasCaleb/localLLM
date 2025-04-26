package org.templateit;

import org.apache.poi.hssf.util.HSSFColor;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.awt.Color;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
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
import com.lowagie.text.Element;
import com.lowagie.text.Font;
import com.lowagie.text.Rectangle;
import com.lowagie.text.pdf.PdfPCell;

@ExtendWith(MockitoExtension.class)
class Poi2ItextUtil_colorPOI2Itext_0_2_Test {

    @Mock
    private HSSFColor mockHSSFColor;

    @Test
    void testColorPOI2Itext() {
        short[] rgb = { 100, 150, 200 };
        when(mockHSSFColor.getTriplet()).thenReturn(rgb);
        Color expectedColor = new Color(rgb[0], rgb[1], rgb[2]);
        Color actualColor = Poi2ItextUtil.colorPOI2Itext(mockHSSFColor);
        assertEquals(expectedColor, actualColor);
        verify(mockHSSFColor, times(1)).getTriplet();
    }

    @Test
    void testColorPOI2ItextWithNull() {
        when(mockHSSFColor.getTriplet()).thenReturn(null);
        assertThrows(NullPointerException.class, () -> Poi2ItextUtil.colorPOI2Itext(mockHSSFColor));
        verify(mockHSSFColor, times(1)).getTriplet();
    }
}
