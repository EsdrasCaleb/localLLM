package org.apache.poi.hssf.usermodel;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.apache.poi.hssf.model.Workbook;
import org.apache.poi.hssf.record.FormatRecord;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class HSSFDataFormat_getFormat_3_4_Test {

    @Mock
    private Workbook workbook;

    @InjectMocks
    private HSSFDataFormat hssfDataFormat;

    @Test
    public void testGetFormatBuiltinFormat() {
        String format = "TEXT";
        when(workbook.getFormat(anyString(), anyBoolean())).thenReturn((short) 0);
        short result = hssfDataFormat.getFormat(format);
        assertEquals(0, result);
    }

    @Test
    public void testGetFormatNewFormat() {
        String format = "NEW_FORMAT";
        when(workbook.getFormat(anyString(), anyBoolean())).thenReturn((short) 1);
        short result = hssfDataFormat.getFormat(format);
        assertEquals(1, result);
    }

    @Test
    public void testGetFormatExistingFormat() {
        String format = "EXISTING_FORMAT";
        when(workbook.getFormat(anyString(), anyBoolean())).thenReturn((short) 2);
        // Add the format to the list
        hssfDataFormat.getFormat(format);
        short result = hssfDataFormat.getFormat(format);
        assertEquals(2, result);
    }

    @Test
    public void testGetFormatMovedBuiltins() throws Exception {
        String format = "MOVED_BUILTINS_FORMAT";
        when(workbook.getFormat(anyString(), anyBoolean())).thenReturn((short) 3);
        // Simulate movedBuiltins being true
        java.lang.reflect.Field movedBuiltinsField = HSSFDataFormat.class.getDeclaredField("movedBuiltins");
        movedBuiltinsField.setAccessible(true);
        movedBuiltinsField.set(hssfDataFormat, true);
        short result = hssfDataFormat.getFormat(format);
        assertEquals(3, result);
    }
}
