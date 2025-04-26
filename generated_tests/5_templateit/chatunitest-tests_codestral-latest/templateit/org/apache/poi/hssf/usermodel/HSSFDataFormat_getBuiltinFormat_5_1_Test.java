package org.apache.poi.hssf.usermodel;

import java.lang.reflect.Field;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.apache.poi.hssf.model.Workbook;
import org.apache.poi.hssf.record.FormatRecord;
import java.util.Iterator;
import java.util.ListIterator;
import java.util.Vector;

class HSSFDataFormat_getBuiltinFormat_5_1_Test {

    private List<String> mockBuiltinFormats;

    @BeforeEach
    void setUp() throws Exception {
        mockBuiltinFormats = mock(List.class);
        Field builtinFormatsField = HSSFDataFormat.class.getDeclaredField("builtinFormats");
        builtinFormatsField.setAccessible(true);
        builtinFormatsField.set(null, mockBuiltinFormats);
    }

    @Test
    void testGetBuiltinFormat() {
        short index = 1;
        String expectedFormat = "mockFormat";
        when(mockBuiltinFormats.get(index)).thenReturn(expectedFormat);
        String result = HSSFDataFormat.getBuiltinFormat(index);
        assertEquals(expectedFormat, result);
        verify(mockBuiltinFormats).get(index);
    }

    @Test
    void testGetBuiltinFormat_IndexOutOfBounds() {
        short index = 100;
        when(mockBuiltinFormats.get(index)).thenThrow(new IndexOutOfBoundsException());
        assertThrows(IndexOutOfBoundsException.class, () -> HSSFDataFormat.getBuiltinFormat(index));
        verify(mockBuiltinFormats).get(index);
    }
}
