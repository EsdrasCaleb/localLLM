package org.templateit;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
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
public class Poi2ItextUtil_colorPOI2Itext_0_2_Test {

    @InjectMocks
    private Poi2ItextUtil poi2ItextUtil;

    @Mock
    private HSSFColor hssfColor;

    @Test
    public void testColorPOI2Itext() {
        // Arrange
        when(hssfColor.getTriplet()).thenReturn(new short[] { 128, 128, 128 });
        // Act
        Color color = Poi2ItextUtil.colorPOI2Itext(hssfColor);
        // Assert
        assertEquals(Color.BLACK, color);
    }
}
