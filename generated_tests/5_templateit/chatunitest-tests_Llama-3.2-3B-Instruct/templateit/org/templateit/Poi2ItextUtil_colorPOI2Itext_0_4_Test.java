package org.templateit;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
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

@ExtendWith(MockitoExtension.class)
@MockitoSettings
public class Poi2ItextUtil_colorPOI2Itext_0_4_Test {

    @Mock
    private HSSFWorkbook workbook;

    @Mock
    private HSSFColor poiColor;

    @InjectMocks
    private Poi2ItextUtil poi2ItextUtil;

    @Test
    public void test_colorPOI2Itext() {
        // Arrange
        // Red color
        when(poiColor.getTriplet()).thenReturn(new short[] { 255, 0, 0 });
        // Act
        Color result = poi2ItextUtil.colorPOI2Itext(poiColor);
        // Assert
        assertEquals(new Color(255, 0, 0), result);
    }

    @Test
    public void test_colorPOI2Itext_Green() {
        // Arrange
        // Green color
        when(poiColor.getTriplet()).thenReturn(new short[] { 0, 255, 0 });
        // Act
        Color result = poi2ItextUtil.colorPOI2Itext(poiColor);
        // Assert
        assertEquals(new Color(0, 255, 0), result);
    }

    @Test
    public void test_colorPOI2Itext_Blue() {
        // Arrange
        // Blue color
        when(poiColor.getTriplet()).thenReturn(new short[] { 0, 0, 255 });
        // Act
        Color result = poi2ItextUtil.colorPOI2Itext(poiColor);
        // Assert
        assertEquals(new Color(0, 0, 255), result);
    }
}
