package org.templateit;

import org.apache.poi.hssf.usermodel.HSSFFont;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import com.lowagie.text.Font;
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
import org.apache.poi.hssf.usermodel.HSSFFormulaEvaluator;
import org.apache.poi.hssf.usermodel.HSSFPalette;
import org.apache.poi.hssf.util.HSSFColor;
import com.lowagie.text.Element;
import com.lowagie.text.Rectangle;
import com.lowagie.text.pdf.PdfPCell;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Poi2ItextUtil_chooseFontFamily_6_2_Test {

    private Poi2ItextUtil poi2ItextUtil;

    private HSSFWorkbook workbook;

    private HSSFFont font;

    @BeforeEach
    public void setUp() {
        workbook = new HSSFWorkbook();
        poi2ItextUtil = new Poi2ItextUtil(workbook);
        font = mock(HSSFFont.class);
    }

    @Test
    public void testChooseFontFamilyArial() {
        when(font.getFontName()).thenReturn("Arial");
        int result = poi2ItextUtil.chooseFontFamily(font, Font.UNDEFINED);
        assertEquals(Font.HELVETICA, result);
    }

    @Test
    public void testChooseFontFamilyCourier() {
        when(font.getFontName()).thenReturn("Courier");
        int result = poi2ItextUtil.chooseFontFamily(font, Font.UNDEFINED);
        assertEquals(Font.COURIER, result);
    }

    @Test
    public void testChooseFontFamilyCourierNew() {
        when(font.getFontName()).thenReturn("Courier New");
        int result = poi2ItextUtil.chooseFontFamily(font, Font.UNDEFINED);
        assertEquals(Font.COURIER, result);
    }

    @Test
    public void testChooseFontFamilyTimesNewRoman() {
        when(font.getFontName()).thenReturn("Times New Roman");
        int result = poi2ItextUtil.chooseFontFamily(font, Font.UNDEFINED);
        assertEquals(Font.TIMES_ROMAN, result);
    }

    @Test
    public void testChooseFontFamilyDefault() {
        when(font.getFontName()).thenReturn("Unknown Font");
        int result = poi2ItextUtil.chooseFontFamily(font, Font.UNDEFINED);
        assertEquals(Font.UNDEFINED, result);
    }
}
