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
public class DynamicTemplate_height_0_1_Test {

    @Mock
    private HSSFSheet sheet;

    @Mock
    private List<NamedStyle> styles;

    @InjectMocks
    private DynamicTemplate dynamicTemplate;

    @Test
    public void testHeight() {
        int expectedHeight = 100;
        when(dynamicTemplate.height()).thenReturn(expectedHeight);
        int actualHeight = dynamicTemplate.height();
        assertEquals(expectedHeight, actualHeight);
    }
}
