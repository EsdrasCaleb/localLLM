package org.apache.poi.hssf.usermodel;

import java.util.List;
import java.util.Vector;
import org.junit.jupiter.api.extension.ExtendWith;
import org.apache.poi.hssf.usermodel.HSSFDataFormat;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.mockito.junit.jupiter.MockitoExtension;
import org.apache.poi.hssf.model.Workbook;
import org.apache.poi.hssf.record.FormatRecord;
import java.util.Iterator;
import java.util.ListIterator;

@ExtendWith(MockitoExtension.class)
public class HSSFDataFormat_getBuiltinFormat_2_0_Test {

    @Mock
    private Workbook workbook;

    private HSSFDataFormat fixture;

    private List<String> builtinFormats;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.initMocks(this);
        fixture = new HSSFDataFormat(workbook);
        builtinFormats = new Vector<>();
        builtinFormats.add("@");
        builtinFormats.add("#,##0");
        builtinFormats.add("$#,##0.00");
        // Add more predefined formats as needed
    }

    @Test
    public void testGetBuiltinFormat() {
        // Mock the workbook to return a predefined format record
        FormatRecord formatRecord = mock(FormatRecord.class);
        when(formatRecord.getFormatString()).thenReturn("1");
        // Call the method correctly
        String result = fixture.getBuiltinFormat((short) 1);
        // Verify the result
        assertEquals("1", result);
    }

    @Test
    public void testGetBuiltinFormatWithNullWorkbook() {
        // Create a new instance of HSSFDataFormat without a workbook
        fixture = new HSSFDataFormat(null);
        // Call the method correctly
        String result = fixture.getBuiltinFormat((short) 1);
        // Verify the result
        assertNull(result);
    }

    @Test
    public void testGetBuiltinFormatText() {
        assertEquals(0, HSSFDataFormat.getBuiltinFormat("text"));
    }

    @Test
    public void testGetBuiltinFormatNumber() {
        assertEquals(1, HSSFDataFormat.getBuiltinFormat("#,##0"));
    }

    @Test
    public void testGetBuiltinFormatCurrency() {
        assertEquals(2, HSSFDataFormat.getBuiltinFormat("$#,##0.00"));
    }

    @Test
    public void testGetBuiltinFormatInvalid() {
        assertEquals(-1, HSSFDataFormat.getBuiltinFormat("invalid"));
    }
}
