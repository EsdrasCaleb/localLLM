package org.templateit;

import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.List;
import org.apache.poi.hssf.usermodel.HSSFCell;

@ExtendWith(MockitoExtension.class)
class DynamicTemplate_getRowHeight_3_0_Test {

    @Mock
    private HSSFSheet sheet;

    @InjectMocks
    private DynamicTemplate template;

    @Test
    void getRowHeight_ValidRow_ReturnsMaxHeight() {
        HSSFRow row1 = mock(HSSFRow.class);
        HSSFRow row2 = mock(HSSFRow.class);
        when(row1.getHeight()).thenReturn((short) 10);
        when(row2.getHeight()).thenReturn((short) 20);
        when(sheet.getRow(0)).thenReturn(row1);
        when(sheet.getRow(1)).thenReturn(row2);
        when(sheet.getRow(2)).thenReturn(null);
        // Initialize DynamicTemplate with mocks
        template = new DynamicTemplate("test", sheet, 2, new ArrayList<>());
        // Test cases
        assertEquals(10, template.getRowHeight(0));
        assertEquals(20, template.getRowHeight(1));
        assertEquals(0, template.getRowHeight(2));
    }

    @Test
    void getRowHeight_EmptyRow_ReturnsZero() {
        when(sheet.getRow(anyInt())).thenReturn(null);
        template = new DynamicTemplate("test", sheet, 2, new ArrayList<>());
        assertEquals(0, template.getRowHeight(0));
    }

    @Test
    void getRowHeight_AllZeroHeights_ReturnsZero() {
        HSSFRow row = mock(HSSFRow.class);
        when(row.getHeight()).thenReturn((short) 0);
        when(sheet.getRow(anyInt())).thenReturn(row);
        template = new DynamicTemplate("test", sheet, 2, new ArrayList<>());
        assertEquals(0, template.getRowHeight(0));
    }
}
