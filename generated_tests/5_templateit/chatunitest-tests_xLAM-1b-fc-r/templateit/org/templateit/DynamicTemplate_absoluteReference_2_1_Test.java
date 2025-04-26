package org.templateit;

import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import java.util.List;
import org.apache.poi.hssf.usermodel.HSSFCell;
import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;

@ExtendWith(MockitoExtension.class)
public class DynamicTemplate_absoluteReference_2_1_Test {

    @InjectMocks
    private DynamicTemplate dynamicTemplate;

    @Test
    public void testAbsoluteReference() {
        int r = 1;
        int c = 2;
        int expectedRow = 10;
        int expectedColumn = 20;
        Reference result = dynamicTemplate.absoluteReference(r, c);
        assertEquals(expectedRow, result.row());
        assertEquals(expectedColumn, result.column());
    }
}
