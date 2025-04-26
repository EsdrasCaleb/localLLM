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
public class DynamicTemplate_getRowHeight_3_0_Test {

    @Mock
    private HSSFSheet sheet;

    @InjectMocks
    private DynamicTemplate dynamicTemplate;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testGetRowHeight() {
        int r = 1;
        int expectedHeight = 50;
        when(sheet.getRow(r)).thenReturn(Mockito.mock(HSSFRow.class));
        when(sheet.getRow(r).getHeight()).thenReturn((short) expectedHeight);
        int actualHeight = dynamicTemplate.getRowHeight(r);
        assertEquals(expectedHeight, actualHeight);
    }
}
