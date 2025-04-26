package org.templateit;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import org.apache.poi.hssf.usermodel.HSSFCell;
import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.List;

@ExtendWith(MockitoExtension.class)
public class DynamicTemplate_getCell_4_1_Test {

    @Mock
    private HSSFSheet sheet;

    private DynamicTemplate dynamicTemplate;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
        dynamicTemplate = new DynamicTemplate("Test", sheet, 10, new ArrayList<>());
    }

    @Test
    public void testGetCell_ValidCoordinates_ReturnsCell() {
        HSSFCell cell = Mockito.mock(HSSFCell.class);
        when(sheet.getRow(5)).thenReturn(Mockito.mock(HSSFRow.class));
        when(sheet.getRow(5).getCell(5)).thenReturn(cell);
        HSSFCell result = dynamicTemplate.getCell(5, 5);
        assertNotNull(result);
    }

    @Test
    public void testGetCell_InvalidCoordinates_ReturnsNull() {
        HSSFCell result = dynamicTemplate.getCell(15, 15);
        assertNull(result);
    }

    @Test
    public void testGetCell_NegativeCoordinates_ReturnsNull() {
        HSSFCell result = dynamicTemplate.getCell(-1, -1);
        assertNull(result);
    }
}
