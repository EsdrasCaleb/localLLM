package org.templateit;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.apache.poi.hssf.usermodel.HSSFCell;
import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;

@ExtendWith(MockitoExtension.class)
public class DynamicTemplate_getCell_4_1_Test {

    @Mock
    private HSSFSheet sheet;

    @InjectMocks
    private DynamicTemplate dynamicTemplate;

    @Test
    void testGetCell_ValidCoordinates() {
        when(sheet.getRow(anyInt())).thenReturn(mock(HSSFRow.class));
        when(dynamicTemplate.width()).thenReturn(10);
        when(dynamicTemplate.height()).thenReturn(10);
        int r = 0;
        int c = 5;
        HSSFCell cell = dynamicTemplate.getCell(r, c);
        Assertions.assertNotNull(cell);
    }

    @Test
    void testGetCell_InvalidCoordinates() {
        when(sheet.getRow(anyInt())).thenReturn(null);
        int r = 10;
        int c = 10;
        HSSFCell cell = dynamicTemplate.getCell(r, c);
        Assertions.assertNull(cell);
    }
}
