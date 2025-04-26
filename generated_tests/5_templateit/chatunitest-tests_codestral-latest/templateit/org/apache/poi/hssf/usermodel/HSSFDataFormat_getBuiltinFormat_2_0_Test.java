package org.apache.poi.hssf.usermodel;

import java.lang.reflect.Field;
import java.util.List;
import java.util.Vector;
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

class HSSFDataFormat_getBuiltinFormat_2_0_Test {

    @BeforeAll
    static void setUp() throws Exception {
        // Mock the builtinFormats list
        List<String> mockBuiltinFormats = new Vector<>();
        for (int i = 0; i <= 0x31; i++) {
            mockBuiltinFormats.add("format" + i);
        }
        // Set the first format to "@" for "TEXT"
        mockBuiltinFormats.set(0, "@");
        // Use reflection to set the builtinFormats field
        Field builtinFormatsField = HSSFDataFormat.class.getDeclaredField("builtinFormats");
        builtinFormatsField.setAccessible(true);
        builtinFormatsField.set(null, mockBuiltinFormats);
    }

    @Test
    void testGetBuiltinFormat() {
        assertEquals(0, HSSFDataFormat.getBuiltinFormat("TEXT"));
        assertEquals(1, HSSFDataFormat.getBuiltinFormat("format1"));
        assertEquals(-1, HSSFDataFormat.getBuiltinFormat("unknownFormat"));
    }
}
