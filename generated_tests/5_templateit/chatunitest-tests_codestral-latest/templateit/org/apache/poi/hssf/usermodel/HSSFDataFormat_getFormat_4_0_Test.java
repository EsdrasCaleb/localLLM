package org.apache.poi.hssf.usermodel;

import java.lang.reflect.Field;
import java.util.List;
import java.util.Vector;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.apache.poi.hssf.model.Workbook;
import org.apache.poi.hssf.record.FormatRecord;
import java.util.Iterator;
import java.util.ListIterator;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class HSSFDataFormat_getFormat_4_0_Test {

    @InjectMocks
    private HSSFDataFormat hssfDataFormat;

    @Mock
    private Workbook workbook;

    @BeforeEach
    public void setUp() throws Exception {
        // Use reflection to set the builtinFormats field
        Field builtinFormatsField = HSSFDataFormat.class.getDeclaredField("builtinFormats");
        builtinFormatsField.setAccessible(true);
        List<String> builtinFormats = new Vector<>();
        builtinFormats.add("format1");
        builtinFormats.add("format2");
        builtinFormatsField.set(hssfDataFormat, builtinFormats);
    }

    @Test
    public void testGetFormat() {
        String format = hssfDataFormat.getFormat((short) 0);
        assertEquals("format1", format);
    }

    @Test
    public void testGetFormatBuiltinsNotMoved() throws Exception {
        // Test when builtins are not moved
        assertEquals("format1", hssfDataFormat.getFormat((short) 0));
        assertEquals("format2", hssfDataFormat.getFormat((short) 1));
    }

    @Test
    public void testGetFormatBuiltinsMoved() throws Exception {
        // Use reflection to set movedBuiltins field to true
        Field movedBuiltinsField = HSSFDataFormat.class.getDeclaredField("movedBuiltins");
        movedBuiltinsField.setAccessible(true);
        movedBuiltinsField.set(hssfDataFormat, true);
        // Use reflection to set formats field
        Field formatsField = HSSFDataFormat.class.getDeclaredField("formats");
        formatsField.setAccessible(true);
        Vector<String> formats = new Vector<>();
        formats.add("movedFormat1");
        formats.add("movedFormat2");
        formatsField.set(hssfDataFormat, formats);
        // Test when builtins are moved
        assertEquals("movedFormat1", hssfDataFormat.getFormat((short) 0));
        assertEquals("movedFormat2", hssfDataFormat.getFormat((short) 1));
    }

    @Test
    public void testGetFormatIndexOutOfBounds() throws Exception {
        // Test when index is out of bounds
        assertThrows(IndexOutOfBoundsException.class, () -> hssfDataFormat.getFormat((short) 10));
    }
}
