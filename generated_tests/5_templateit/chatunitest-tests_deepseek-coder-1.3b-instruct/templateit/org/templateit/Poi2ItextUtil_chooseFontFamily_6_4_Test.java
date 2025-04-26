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

class Poi2ItextUtil_chooseFontFamily_6_4_Test {

    @Test
    void chooseFontFamilyTest() {
        HSSFFont mockedFont = Mockito.mock(HSSFFont.class);
        Mockito.when(mockedFont.getFontName()).thenReturn("Arial", "Courier", "Courier New", "Times New Roman");
        Poi2ItextUtil poi2ItextUtil = new Poi2ItextUtil(null);
        int defaultFontFamily = 0;
        assertEquals(0, poi2ItextUtil.chooseFontFamily(mockedFont, defaultFontFamily));
        Mockito.verify(mockedFont, Mockito.times(4)).getFontName();
    }
}
