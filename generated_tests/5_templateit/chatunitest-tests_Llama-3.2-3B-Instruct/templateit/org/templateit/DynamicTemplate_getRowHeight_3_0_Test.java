package org.templateit;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
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

    @Mock
    private List<NamedStyle> styles;

    private DynamicTemplate dynamicTemplate;

    @BeforeEach
    public void setup() {
        dynamicTemplate = new DynamicTemplate(sheet, styles);
    }

    @Test
    public void testGetRowHeight() {
        dynamicTemplate.setHeight(10);
        dynamicTemplate.setWidth(5);
        List<NamedStyle> namedStyles = new ArrayList<>();
        when(styles).thenReturn(namedStyles);
        // Act
        int result = dynamicTemplate.getRowHeight(0);
        // Assert
        assertEquals(10, result);
    }

    @Test
    public void testGetRowHeight_NoRows() {
        dynamicTemplate.setHeight(10);
        dynamicTemplate.setWidth(5);
        List<NamedStyle> namedStyles = new ArrayList<>();
        when(styles).thenReturn(namedStyles);
        // Act
        int result = dynamicTemplate.getRowHeight(5);
        // Assert
        assertEquals(0, result);
    }

    @Test
    public void testGetRowHeight_MultipleRows() {
        dynamicTemplate.setHeight(10);
        dynamicTemplate.setWidth(5);
        List<NamedStyle> namedStyles = new ArrayList<>();
        NamedStyle style1 = new NamedStyle("style1", true);
        NamedStyle style2 = new NamedStyle("style2", false);
        namedStyles.add(style1);
        namedStyles.add(style2);
        when(styles).thenReturn(namedStyles);
        // Act
        int result = dynamicTemplate.getRowHeight(1);
        // Assert
        assertEquals(10, result);
    }

    @Test
    public void testGetRowHeight_NullRow() {
        dynamicTemplate.setHeight(10);
        dynamicTemplate.setWidth(5);
        List<NamedStyle> namedStyles = new ArrayList<>();
        when(styles).thenReturn(namedStyles);
        // Act and Assert
        assertThrows(NullPointerException.class, () -> dynamicTemplate.getRowHeight(10));
    }
}

class DynamicTemplate {

    private final HSSFSheet sheet;

    private final List<NamedStyle> styles;

    private int height;

    private int width;

    public DynamicTemplate(HSSFSheet sheet, List<NamedStyle> styles) {
        this.sheet = sheet;
        this.styles = styles;
    }

    public void setHeight(int height) {
        this.height = height;
    }

    public void setWidth(int width) {
        this.width = width;
    }

    public int getRowHeight(int row) {
        // logic to calculate row height
        // for simplicity, return height
        return height;
    }
}

class NamedStyle {

    private String name;

    private boolean isApplied;

    public NamedStyle(String name, boolean isApplied) {
        this.name = name;
        this.isApplied = isApplied;
    }
}
