package org.templateit;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.List;
import org.apache.poi.hssf.usermodel.HSSFCell;
import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class DynamicTemplate_height_0_1_Test {

    @Mock
    private HSSFSheet sheet;

    @Mock
    private List<NamedStyle> styles;

    private DynamicTemplate dynamicTemplate;

    @BeforeEach
    public void setup() {
        dynamicTemplate = new DynamicTemplate(null, sheet, 10, styles);
    }

    @Test
    public void testHeight() {
        // Set the styles
        when(styles.size()).thenReturn(10);
        // Set the sheet
        when(sheet.getRow(0)).thenReturn(null);
        when(sheet.getRow(1)).thenReturn(null);
        when(sheet.getRow(2)).thenReturn(null);
        when(sheet.getRow(3)).thenReturn(null);
        when(sheet.getRow(4)).thenReturn(null);
        when(sheet.getRow(5)).thenReturn(null);
        when(sheet.getRow(6)).thenReturn(null);
        when(sheet.getRow(7)).thenReturn(null);
        when(sheet.getRow(8)).thenReturn(null);
        when(sheet.getRow(9)).thenReturn(null);
        // Get the height
        int height = dynamicTemplate.height();
        // Verify the height
        assertEquals(10, height);
    }
}
