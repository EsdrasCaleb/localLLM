package org.templateit;

import org.apache.poi.hssf.usermodel.*;
import java.awt.Color;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.awt.font.FontRenderContext;
import java.awt.font.TextAttribute;
import java.awt.font.TextLayout;
import java.text.AttributedString;
import org.apache.log4j.Logger;
import org.apache.poi.hssf.util.HSSFColor;
import com.lowagie.text.Element;
import com.lowagie.text.Font;
import com.lowagie.text.Rectangle;
import com.lowagie.text.pdf.PdfPCell;

public class Poi2ItextUtil_colorPOI2Itext_0_0_Test {

    @Test
    public void testColorPOI2Itext() {
        HSSFColor poiColor = Mockito.mock(HSSFColor.class);
        short[] poiRGB = { 1, 2, 3 };
        Mockito.when(poiColor.getTriplet()).thenReturn(poiRGB);
        Color expectedColor = new Color(1, 2, 3);
        Color result = Poi2ItextUtil.colorPOI2Itext(poiColor);
        assertEquals(expectedColor, result);
    }
}
