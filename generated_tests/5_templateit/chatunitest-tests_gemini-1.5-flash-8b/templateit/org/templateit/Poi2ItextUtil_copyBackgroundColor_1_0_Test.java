package org.templateit;

import org.apache.poi.hssf.usermodel.*;
import org.apache.poi.hssf.util.HSSFColor;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.awt.*;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.awt.font.FontRenderContext;
import java.awt.font.TextAttribute;
import java.awt.font.TextLayout;
import java.text.AttributedString;
import org.apache.log4j.Logger;
import com.lowagie.text.Element;
import com.lowagie.text.Font;
import com.lowagie.text.Rectangle;
import com.lowagie.text.pdf.PdfPCell;

@ExtendWith(MockitoExtension.class)
class Poi2ItextUtil_copyBackgroundColor_1_0_Test {

    @Mock
    private HSSFWorkbook workbook;

    @Mock
    private HSSFCell xcell;

    @Mock
    private PdfPCell cell;

    @InjectMocks
    private Poi2ItextUtil poi2ItextUtil;

    @BeforeEach
    void setUp() {
        // No need to create a new Poi2ItextUtil instance here.
        // Mockito handles injection of mocks.
    }

    @Test
    void copyBackgroundColor_shouldSetBackgroundColorIfColorAvailable() {
        // Arrange
        HSSFCellStyle style = mock(HSSFCellStyle.class);
        when(xcell.getCellStyle()).thenReturn(style);
        short color = HSSFColor.BLACK.index;
        when(style.getFillForegroundColor()).thenReturn(color);
        when(workbook.getCustomPalette()).thenReturn(mock(HSSFPalette.class));
        // Act
        poi2ItextUtil.copyBackgroundColor(xcell, cell);
        // Correctly verify with the Color type
        verify(cell).setBackgroundColor(new Color(color));
    }

    @Test
    void copyBackgroundColor_shouldNotSetBackgroundColorIfColorNotAvailable() {
        // Arrange
        HSSFCellStyle style = mock(HSSFCellStyle.class);
        when(xcell.getCellStyle()).thenReturn(style);
        when(style.getFillForegroundColor()).thenReturn((short) -1);
        when(workbook.getCustomPalette()).thenReturn(mock(HSSFPalette.class));
        // Act
        poi2ItextUtil.copyBackgroundColor(xcell, cell);
        // Correctly verify with the Color type
        verify(cell, never()).setBackgroundColor(any(Color.class));
    }

    @Test
    void copyBackgroundColor_validColor() {
        HSSFCellStyle xstyle = Mockito.mock(HSSFCellStyle.class);
        HSSFColor poiColor = Mockito.mock(HSSFColor.class);
        when(xcell.getCellStyle()).thenReturn(xstyle);
        when(xstyle.getFillForegroundColor()).thenReturn((short) 1);
        when(workbook.getCustomPalette()).thenReturn(Mockito.mock(HSSFPalette.class));
        when(poiColor.getIndex()).thenReturn((short) 1);
        when(workbook.getCustomPalette().getColor((short) 1)).thenReturn(poiColor);
        when(poi2ItextUtil.colorPOI2Itext(poiColor)).thenReturn(Color.RED);
        poi2ItextUtil.copyBackgroundColor(xcell, cell);
        verify(cell).setBackgroundColor(Color.RED);
    }

    @Test
    void copyBackgroundColor_invalidColor() {
        HSSFCellStyle xstyle = Mockito.mock(HSSFCellStyle.class);
        HSSFColor poiColor = Mockito.mock(HSSFColor.class);
        when(xcell.getCellStyle()).thenReturn(xstyle);
        when(xstyle.getFillForegroundColor()).thenReturn((short) 1);
        when(workbook.getCustomPalette()).thenReturn(Mockito.mock(HSSFPalette.class));
        when(poiColor.getIndex()).thenReturn(HSSFColor.AUTOMATIC.index);
        when(workbook.getCustomPalette().getColor((short) 1)).thenReturn(poiColor);
        poi2ItextUtil.copyBackgroundColor(xcell, cell);
        verify(cell, never()).setBackgroundColor(any(Color.class));
    }
}
