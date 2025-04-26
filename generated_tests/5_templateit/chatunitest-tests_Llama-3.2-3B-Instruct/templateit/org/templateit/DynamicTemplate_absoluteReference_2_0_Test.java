package org.templateit;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.apache.poi.hssf.usermodel.HSSFCell;
import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;

@ExtendWith(MockitoExtension.class)
public class DynamicTemplate_absoluteReference_2_0_Test {

    @Mock
    private List<NamedStyle> styles;

    @InjectMocks
    private DynamicTemplate dynamicTemplate;

    @Test
    public void testAbsoluteReference() throws Exception {
        // Arrange
        when(styles.size()).thenReturn(10);
        when(styles.get(0).getRow()).thenReturn(1);
        when(styles.get(0).getColumn()).thenReturn(2);
        when(styles.get(1).getRow()).thenReturn(3);
        when(styles.get(1).getColumn()).thenReturn(4);
        when(styles.get(9).getRow()).thenReturn(10);
        when(styles.get(9).getColumn()).thenReturn(20);
        // Act
        Method method = DynamicTemplate.class.getDeclaredMethod("absoluteReference", int.class, int.class);
        method.setAccessible(true);
        Reference reference1 = (Reference) method.invoke(dynamicTemplate, 0, 0);
        Reference reference2 = (Reference) method.invoke(dynamicTemplate, 1, 1);
        Reference reference3 = (Reference) method.invoke(dynamicTemplate, 9, 9);
        // Assert
        assertEquals(new Reference(1, 2), reference1);
        assertEquals(new Reference(3, 4), reference2);
        assertEquals(new Reference(10, 20), reference3);
    }
}
