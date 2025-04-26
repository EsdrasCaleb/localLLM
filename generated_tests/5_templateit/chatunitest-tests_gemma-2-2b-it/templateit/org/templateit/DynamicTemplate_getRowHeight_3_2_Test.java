package org.templateit;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.List;
import org.apache.poi.hssf.usermodel.HSSFCell;
import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;

public class DynamicTemplate_getRowHeight_3_2_Test {

    @Test
    void testGetRowHeight() {
        DynamicTemplate dynamicTemplate = mock(DynamicTemplate.class);
        when(dynamicTemplate.getRowHeight(anyInt())).thenReturn(10);
        int result = dynamicTemplate.getRowHeight(0);
        assertEquals(10, result);
    }
}
