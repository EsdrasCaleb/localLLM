package org.templateit;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.List;
import org.apache.poi.hssf.usermodel.HSSFCell;
import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;

@ExtendWith(MockitoExtension.class)
class DynamicTemplate_height_0_2_Test {

    @Mock
    DynamicTemplate template;

    @InjectMocks
    DynamicTemplate_height_0_2_Test test;

    @Test
    void testHeight() {
        when(template.height()).thenReturn(10);
        assertEquals(10, test.template.height());
    }
}
