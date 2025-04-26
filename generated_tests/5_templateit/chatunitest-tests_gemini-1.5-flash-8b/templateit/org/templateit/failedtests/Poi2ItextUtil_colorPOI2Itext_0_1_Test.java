package org.templateit;

import org.apache.poi.hssf.util.HSSFColor;
import java.awt.Color;
import org.mockito.junit.jupiter.MockitoExtension;
import org.junit.jupiter.api.extension.ExtendWith;
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
class Poi2ItextUtil_colorPOI2Itext_0_1_Test {

    @Test
    void testColorPOI2Itext_validInput() {
        // Arrange
        short[] rgb = { 100, 200, 50 };
        HSSFColor mockColor = Mockito.mock(HSSFColor.class);
        Mockito.when(mockColor.getTriplet()).thenReturn(rgb);
        // Act
        Color result = Poi2ItextUtil.colorPOI2Itext(mockColor);
        // Assert
        Assertions.assertEquals(new Color(100, 200, 50), result);
    }

    @Test
    void testColorPOI2Itext_nullInput() {
        // Arrange
        HSSFColor mockColor = null;
        // Act & Assert (expecting no exception, but potentially null return)
        Color result = Poi2ItextUtil.colorPOI2Itext(mockColor);
        Assertions.assertNull(result);
    }

    @Test
    void testColorPOI2Itext_emptyInput() {
        // Arrange
        short[] rgb = {};
        HSSFColor mockColor = Mockito.mock(HSSFColor.class);
        Mockito.when(mockColor.getTriplet()).thenReturn(rgb);
        // Act
        Color result = Poi2ItextUtil.colorPOI2Itext(mockColor);
        // Assert - important to check for unexpected behavior
        Assertions.assertNull(result);
    }

    @Test
    void testColorPOI2Itext_invalidInput() {
        // Arrange
        // Invalid RGB value (exceeds maximum)
        short[] rgb = { 256, 256, 256 };
        HSSFColor mockColor = Mockito.mock(HSSFColor.class);
        Mockito.when(mockColor.getTriplet()).thenReturn(rgb);
        // Act
        Color result = Poi2ItextUtil.colorPOI2Itext(mockColor);
        // Assert - Expecting clamping to maximum value (255)
        Assertions.assertEquals(new Color(255, 255, 255), result);
    }
}
