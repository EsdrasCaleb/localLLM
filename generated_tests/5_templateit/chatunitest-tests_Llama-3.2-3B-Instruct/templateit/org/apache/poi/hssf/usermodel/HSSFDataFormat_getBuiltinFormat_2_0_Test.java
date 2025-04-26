package org.apache.poi.hssf.usermodel;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.apache.poi.hssf.model.Workbook;
import org.apache.poi.hssf.record.FormatRecord;
import java.util.Iterator;
import java.util.ListIterator;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class HSSFDataFormat_getBuiltinFormat_2_0_Test {

    @Mock
    private Workbook workbook;

    @InjectMocks
    private HSSFDataFormat hssfDataFormat;

    @Test
    public void testGetBuiltinFormat_TextFormat_ReturnsCorrectIndex() {
        // Arrange
        List<String> builtinFormats = new ArrayList<>();
        builtinFormats.add("TEXT");
        builtinFormats.add("OTHER");
        when(hssfDataFormat.getBuiltinFormats()).thenReturn(builtinFormats);
        // Act
        short index = hssfDataFormat.getBuiltinFormat("TEXT");
        // Assert
        assertEquals(0, index);
    }

    @Test
    public void testGetBuiltinFormat_OtherFormat_ReturnsMinusOne() {
        // Arrange
        List<String> builtinFormats = new ArrayList<>();
        when(hssfDataFormat.getBuiltinFormats()).thenReturn(builtinFormats);
        // Act
        short index = hssfDataFormat.getBuiltinFormat("OTHER");
        // Assert
        assertEquals(-1, index);
    }

    @Test
    public void testGetBuiltinFormat_NullFormat_ThrowsNullPointerException() {
        // Act and Assert
        assertThrows(NullPointerException.class, () -> hssfDataFormat.getBuiltinFormat(null));
    }

    @Test
    public void testGetBuiltinFormat_EmptyFormat_ThrowsNullPointerException() {
        // Act and Assert
        assertThrows(NullPointerException.class, () -> hssfDataFormat.getBuiltinFormat(""));
    }
}
