package org.templateit;

import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.templateit.DynamicTemplate;
import org.templateit.NamedStyle;
import java.util.ArrayList;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.apache.poi.hssf.usermodel.HSSFCell;
import org.apache.poi.hssf.usermodel.HSSFRow;

@ExtendWith(MockitoExtension.class)
class DynamicTemplate_height_0_2_Test {

    @Mock
    private HSSFSheet sheet;

    @InjectMocks
    private DynamicTemplate template;

    @Test
    void testHeight() {
        // Arrange
        int expectedHeight = 10;
        List<NamedStyle> styles = new ArrayList<>();
        // Crucial: Mock the sheet's getLastRowNum method
        // Adjust for 0-based index
        when(sheet.getLastRowNum()).thenReturn(expectedHeight - 1);
        // Now create the template with the mocked sheet
        template = new DynamicTemplate("test", sheet, expectedHeight, styles);
        // Act
        int actualHeight = template.height();
        // Assert
        assertEquals(expectedHeight, actualHeight);
    }

    @Test
    void testHeightWithZero() {
        // Arrange
        int expectedHeight = 0;
        List<NamedStyle> styles = new ArrayList<>();
        // Adjust for 0-based index
        when(sheet.getLastRowNum()).thenReturn(-1);
        template = new DynamicTemplate("test", sheet, expectedHeight, styles);
        // Act
        int actualHeight = template.height();
        // Assert
        assertEquals(expectedHeight, actualHeight);
    }
}
